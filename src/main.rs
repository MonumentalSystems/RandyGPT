mod checkpoint;
mod config;
mod forward;
mod metal;
mod model;
mod ops;
mod optimizer;
mod rng;
mod tokenizer;
mod train;

use std::fs::File;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use checkpoint::{load_checkpoint, load_checkpoint_cpu, load_checkpoint_v2, load_checkpoint_v3};
use config::*;
use metal::METAL_DEVICE;
use model::{CandleModel, GPTModel};
use optimizer::GpuAdamState;
use rng::Rng;
use tokenizer::Tokenizer;
use train::{estimate_loss, generate, generate_cpu, train, train_candle};

fn load_training_data(path: &str) -> std::io::Result<String> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut text = String::new();
    for line in reader.lines() {
        text.push_str(&line?);
        text.push('\n');
    }
    Ok(text)
}

fn main() -> std::io::Result<()> {
    // ── CLI arguments ─────────────────────────────────────────────────
    // Usage: randygpt [--iters N] [--resume [path]]
    let args: Vec<String> = std::env::args().collect();
    let mut iterations = MAX_ITERS;
    let mut resume_path: Option<String> = None;
    let mut lr_override:     Option<f32> = None;
    let mut min_lr_override: Option<f32> = None;
    let mut bpe_vocab_size:  Option<usize> = None;
    let mut generate_mode:   bool = false;
    let mut generate_prompts: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                i += 1;
                if i < args.len() {
                    iterations = args[i].parse().unwrap_or(MAX_ITERS);
                }
            }
            "--resume" => {
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    resume_path = Some(args[i].clone());
                } else {
                    resume_path = Some("checkpoint.bin".to_string());
                }
            }
            "--lr" => {
                i += 1;
                if i < args.len() {
                    lr_override = args[i].parse().ok();
                }
            }
            "--min-lr" => {
                i += 1;
                if i < args.len() {
                    min_lr_override = args[i].parse().ok();
                }
            }
            "--bpe" => {
                // --bpe        → use default BPE_VOCAB_SIZE
                // --bpe 3000   → use custom target vocab size
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    bpe_vocab_size = Some(args[i].parse().unwrap_or(BPE_VOCAB_SIZE));
                } else {
                    bpe_vocab_size = Some(BPE_VOCAB_SIZE);
                }
            }
            "--generate" => {
                // --generate                     → use default prompts
                // --generate "prompt1" "prompt2"  → use custom prompts
                generate_mode = true;
                while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    generate_prompts.push(args[i].clone());
                }
            }
            other => {
                if let Ok(n) = other.parse::<usize>() {
                    iterations = n;
                } else {
                    eprintln!("Unknown argument '{}'. Ignoring.", other);
                }
            }
        }
        i += 1;
    }
    let lr     = lr_override.unwrap_or(LEARNING_RATE);
    let min_lr = min_lr_override.unwrap_or(MIN_LEARNING_RATE);

    // --generate implies --resume if no explicit --resume given
    if generate_mode && resume_path.is_none() {
        if Path::new("checkpoint_best.bin").exists() {
            resume_path = Some("checkpoint_best.bin".to_string());
        } else if Path::new("checkpoint.bin").exists() {
            resume_path = Some("checkpoint.bin".to_string());
        } else {
            eprintln!("Error: --generate requires a checkpoint file. Train first or specify --resume <path>.");
            return Ok(());
        }
    }

    if !generate_mode && resume_path.is_none() && Path::new("checkpoint.bin").exists() {
        eprintln!("Found checkpoint.bin — use --resume to continue from it, or delete it to start fresh.");
    }
    if lr_override.is_some() || min_lr_override.is_some() {
        println!("LR override: {} → {}", lr, min_lr);
    }

    let model_size_name = if cfg!(feature = "model-xs")   { "XS (~726K)"   }
                          else if cfg!(feature = "model-s")    { "S (~1.6M)"    }
                          else if cfg!(feature = "model-m")    { "M (~2.7M)"    }
                          else if cfg!(feature = "model-deep") { "Deep (~7.5M)" }
                          else if cfg!(feature = "model-xl")   { "XL (~10.8M)"  }
                          else                                 { "L (~4.82M)"   };
    println!("=== Enhanced randyGPT ===");
    println!("Model: {} — {} layers, {} heads, {}-dim", model_size_name, N_LAYER, N_HEAD, N_EMBD);
    println!("Block size: {}, Vocab size: up to {}", BLOCK_SIZE, MAX_VOCAB);
    println!();

    let mut rng = Rng::new(1337);

    // ── Generate-only: skip training data, just load tokenizer ──────
    if generate_mode {
        let tokenizer = if let Some(_target) = bpe_vocab_size {
            if Path::new(BPE_VOCAB_PATH).exists() {
                println!("Loading BPE vocab from {}...", BPE_VOCAB_PATH);
                let t = Tokenizer::load_bpe(BPE_VOCAB_PATH)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                println!("Loaded BPE vocab ({} tokens)", t.vocab_size);
                t
            } else {
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                    "No vocab.json found. Train a model first before using --generate."));
            }
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::Other,
                "--generate requires BPE mode (--bpe N). Char-level generate needs training data."));
        };

        println!("Vocabulary size: {}", tokenizer.vocab_size);
        println!();

        // Load model + checkpoint
        let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);
        let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len()
            + N_LAYER * (
                model.layers[0].wq.len() + model.layers[0].wk.len()
                + model.layers[0].wv.len() + model.layers[0].wo.len()
                + model.layers[0].fc1.len() + model.layers[0].fc2.len()
            );
        println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);

        if let Some(ref path) = resume_path {
            println!("Loading checkpoint: {}...", path);
            load_checkpoint_cpu(path, &mut model)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                "No checkpoint found for --generate. Train a model first."));
        }

        let prompts: Vec<&str> = if generate_prompts.is_empty() {
            vec!["The ", "Once upon a time", "He said", "She walked into the room", "Chapter 3"]
        } else {
            generate_prompts.iter().map(|s| s.as_str()).collect()
        };
        println!("=== Generation Mode ===");
        println!("Checkpoint: {}", resume_path.as_deref().unwrap_or("?"));
        println!();
        for prompt in &prompts {
            println!("────────────────────────────────────");
            println!("Prompt: \"{}\"", prompt);
            println!("────────────────────────────────────");
            let sample = generate_cpu(&model, &tokenizer, prompt, 200, 0.8, 0.9, &mut rng);
            println!("{}", sample);
            println!();
        }
        return Ok(());
    }

    // ── Load and split training data ──────────────────────────────────
    let training_text = if Path::new("train.txt").exists() {
        println!("Loading training data from train.txt...");
        load_training_data("train.txt")?
    } else {
        println!("No train.txt found. Using default sample data.");
        concat!(
            "The quick brown fox jumps over the lazy dog. ",
            "Rust is a systems programming language. ",
            "Machine learning models learn from data. ",
            "Transformers use attention mechanisms. ",
            "GPT stands for Generative Pre-trained Transformer. ",
            "Neural networks are inspired by the human brain. ",
            "Deep learning is a subset of machine learning. "
        ).to_string()
    };

    println!("Training data size: {} characters", training_text.len());

    // ── Build or load tokenizer ───────────────────────────────────────
    let tokenizer = if let Some(target) = bpe_vocab_size {
        if Path::new(BPE_VOCAB_PATH).exists() {
            println!("Loading BPE vocab from {}...", BPE_VOCAB_PATH);
            match Tokenizer::load_bpe(BPE_VOCAB_PATH) {
                Ok(t)  => { println!("Loaded BPE vocab ({} tokens)", t.vocab_size); t }
                Err(e) => {
                    eprintln!("Failed to load {}: {}. Retraining...", BPE_VOCAB_PATH, e);
                    let t = Tokenizer::from_text_bpe(&training_text, target);
                    t.save_bpe(BPE_VOCAB_PATH)?;
                    println!("BPE vocab ({} tokens) saved to {}", t.vocab_size, BPE_VOCAB_PATH);
                    t
                }
            }
        } else {
            println!("Training BPE tokenizer (target vocab: {})...", target);
            let t = Tokenizer::from_text_bpe(&training_text, target);
            t.save_bpe(BPE_VOCAB_PATH)?;
            println!("BPE vocab ({} tokens) saved to {}", t.vocab_size, BPE_VOCAB_PATH);
            t
        }
    } else {
        Tokenizer::from_text(&training_text)
    };

    println!("Vocabulary size: {}", tokenizer.vocab_size);
    println!("Sample tokens: {:?}", tokenizer.sample_tokens(10));
    println!();

    // ── Tokenize (with binary cache) ──────────────────────────────────
    let token_cache_path = "tokens.bin";
    let data_all = if Path::new(token_cache_path).exists() {
        println!("Loading cached tokens from {}...", token_cache_path);
        let mut f = File::open(token_cache_path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        // tokens stored as u32 little-endian
        let tokens: Vec<usize> = buf.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        println!("Loaded {} cached tokens", tokens.len());
        tokens
    } else {
        println!("Tokenizing text ({} chars)...", training_text.len());
        let tokens = tokenizer.encode(&training_text);
        // Save cache
        let mut f = File::create(token_cache_path)?;
        for &t in &tokens {
            f.write_all(&(t as u32).to_le_bytes())?;
        }
        println!("Saved token cache to {} ({:.1}MB)",
            token_cache_path, (tokens.len() * 4) as f64 / 1_048_576.0);
        tokens
    };

    let val_split = (data_all.len() * 9) / 10;
    let data     = data_all[..val_split].to_vec();
    let val_data = data_all[val_split..].to_vec();
    println!("Tokenized to {} tokens ({} train, {} val)",
        data_all.len(), data.len(), val_data.len());

    // ── Initialize model ──────────────────────────────────────────────
    println!("Initializing model...");
    let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);

    let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len()
        + N_LAYER * (
            model.layers[0].wq.len() + model.layers[0].wk.len()
            + model.layers[0].wv.len() + model.layers[0].wo.len()
            + model.layers[0].fc1.len() + model.layers[0].fc2.len()
        );
    println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);
    println!();

    // ── Force Metal init so we know which path to take ───────────────
    let use_metal = METAL_DEVICE.is_some();
    if use_metal {
        println!("Metal GPU: enabled — training via Candle autograd");
    } else {
        println!("Metal GPU: unavailable — training on CPU (BLAS)");
    }
    println!();

    // ── Resume from checkpoint ────────────────────────────────────────
    // On Metal: try RGPT0003, then RGPT0002 (moments reset), then RGPT0001.
    // On CPU:   RGPT0001 only.
    //
    // candle_resume holds (CandleModel, GpuAdamState, iter, step, best_loss)
    // if an RGPT0003 checkpoint was successfully loaded; otherwise None and
    // the model weights are available in the CPU `model` variable.
    let mut candle_resume: Option<(CandleModel, GpuAdamState, usize, usize, f32)> = None;

    let (iter_start, step_start, best_loss_start) = if let Some(ref ckpt) = resume_path {
        let result: std::io::Result<(usize, usize, f32)> = if use_metal {
            let device = METAL_DEVICE.as_ref().unwrap();

            // Try RGPT0003 (full GPU state)
            let r3 = {
                let mut cm = CandleModel::from_gpt(&model, device)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                let vars = cm.all_vars();
                let mut opt = GpuAdamState::new(&vars)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                load_checkpoint_v3(ckpt, &mut cm, &mut opt).map(|(it, st, bl)| {
                    candle_resume = Some((cm, opt, it, st, bl));
                    (it, st, bl)
                })
            };

            if r3.is_ok() {
                r3
            } else {
                // Try RGPT0002 (weights only, moments reset to zero)
                let r2 = {
                    let mut cm = CandleModel::from_gpt(&model, device)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                    load_checkpoint_v2(ckpt, &mut cm).map(|(it, st, bl)| {
                        let vars = cm.all_vars();
                        let opt = GpuAdamState::new(&vars)
                            .expect("GpuAdamState init failed");
                        candle_resume = Some((cm, opt, it, st, bl));
                        (it, st, bl)
                    })
                };
                if r2.is_ok() { r2 } else { load_checkpoint(ckpt, &mut model) }
            }
        } else {
            load_checkpoint(ckpt, &mut model)
        };

        match result {
            Ok((it, st, bl)) => {
                println!("✓ Resumed from '{}' — iter {}, step {}, best loss {:.4}", ckpt, it, st, bl);
                println!();
                (it, st, bl)
            }
            Err(e) => {
                eprintln!("Error loading checkpoint '{}': {}", ckpt, e);
                eprintln!("Starting from scratch instead.");
                (0, 0, f32::INFINITY)
            }
        }
    } else {
        (0, 0, f32::INFINITY)
    };

    // ── Sync resumed weights to CPU model ──────────────────────────
    // When resuming on Metal, the loaded weights live in the CandleModel.
    // Sync them back to `model` now so estimate_loss / generate sees the
    // actual checkpoint state, not freshly-initialized random weights.
    if let Some((ref cm, _, _, _, _)) = candle_resume {
        model = cm.to_gpt()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    }

    if iter_start >= iterations {
        println!("Already at iteration {} (target {}). Nothing to train.", iter_start, iterations);
        println!("Increase --iters to continue training.");
        return Ok(());
    }

    // ── Initial loss estimate ─────────────────────────────────────────
    println!("Estimating initial loss...");
    let initial_loss     = estimate_loss(&model, &data, 50, &mut rng);
    let initial_val_loss = estimate_loss(&model, &val_data, 50, &mut rng);
    println!("Initial loss: {:.4} | Val: {:.4} (ppl {:.1})",
        initial_loss, initial_val_loss, initial_val_loss.exp());
    println!();

    // ── Ctrl-C handler ────────────────────────────────────────────────
    let ctrlc_flag = Arc::new(AtomicBool::new(false));
    {
        let flag = ctrlc_flag.clone();
        ctrlc::set_handler(move || { flag.store(true, Ordering::Relaxed); })
            .expect("Error setting Ctrl-C handler");
    }

    // ── Train ─────────────────────────────────────────────────────────
    if use_metal {
        let device = METAL_DEVICE.as_ref().unwrap();
        let (mut candle_model, mut opt) = if let Some((cm, o, _, _, _)) = candle_resume {
            (cm, o)
        } else {
            let cm = CandleModel::from_gpt(&model, device)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            let vars = cm.all_vars();
            let o = GpuAdamState::new(&vars)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            (cm, o)
        };
        // Sync step_t so bias correction starts correctly
        opt.step_t = step_start;
        train_candle(&mut candle_model, &mut opt, &data, &val_data, iterations, &mut rng,
            iter_start, step_start, best_loss_start, lr, min_lr, ctrlc_flag);
        model = candle_model.to_gpt()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    } else {
        train(&mut model, &data, &val_data, iterations, &mut rng,
            iter_start, step_start, best_loss_start, lr, min_lr, ctrlc_flag);
    }

    // ── Final loss estimate ───────────────────────────────────────────
    println!("Estimating final loss...");
    let final_loss     = estimate_loss(&model, &data, 50, &mut rng);
    let final_val_loss = estimate_loss(&model, &val_data, 50, &mut rng);
    println!("Final train loss: {:.4} (started {:.4})", final_loss, initial_loss);
    println!("Final val loss:   {:.4} (ppl {:.1}, started {:.4})",
        final_val_loss, final_val_loss.exp(), initial_val_loss);
    println!();

    // ── Generate samples ──────────────────────────────────────────────
    println!("=== Generation After Training ===");
    for (prompt, max_tokens) in &[("ROMEO:", 100), ("To be or not to be", 100), ("Once upon a time", 100)] {
        println!("\nPrompt: \"{}\"", prompt);
        let sample = generate(&model, &tokenizer, prompt, *max_tokens, 0.8, 0.9, &mut rng);
        println!("{}", sample);
    }

    Ok(())
}
