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
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use checkpoint::load_checkpoint;
use config::*;
use model::GPTModel;
use rng::Rng;
use tokenizer::Tokenizer;
use train::{estimate_loss, generate, train};

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

    if resume_path.is_none() && Path::new("checkpoint.bin").exists() {
        eprintln!("Found checkpoint.bin — use --resume to continue from it, or delete it to start fresh.");
    }

    println!("=== Enhanced randyGPT ===");
    println!("Model: {} layers, {} heads, {} embedding dim", N_LAYER, N_HEAD, N_EMBD);
    println!("Block size: {}, Vocab size: up to {}", BLOCK_SIZE, MAX_VOCAB);
    println!();

    let mut rng = Rng::new(1337);

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

    let tokenizer = Tokenizer::from_text(&training_text);
    println!("Vocabulary size: {}", tokenizer.vocab_size);
    println!("Sample tokens: {:?}", &tokenizer.idx_to_char[..10.min(tokenizer.vocab_size)]);
    println!();

    // 90/10 train/val split
    let data_all = tokenizer.encode(&training_text);
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

    // ── Resume from checkpoint ────────────────────────────────────────
    let (iter_start, step_start, best_loss_start) = if let Some(ref ckpt) = resume_path {
        match load_checkpoint(ckpt, &mut model) {
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

    if iter_start >= iterations {
        println!("Already at iteration {} (target {}). Nothing to train.", iter_start, iterations);
        println!("Increase --iters to continue training.");
        return Ok(());
    }

    // ── Initial loss estimate ─────────────────────────────────────────
    println!("Estimating initial loss...");
    let initial_loss     = estimate_loss(&model, &data, 10, &mut rng);
    let initial_val_loss = estimate_loss(&model, &val_data, 10, &mut rng);
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
    train(&mut model, &data, &val_data, iterations, &mut rng,
        iter_start, step_start, best_loss_start, ctrlc_flag);

    // ── Final loss estimate ───────────────────────────────────────────
    println!("Estimating final loss...");
    let final_loss     = estimate_loss(&model, &data, 10, &mut rng);
    let final_val_loss = estimate_loss(&model, &val_data, 10, &mut rng);
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
