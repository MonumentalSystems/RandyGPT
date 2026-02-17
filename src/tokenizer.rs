/* ------------------------------------------------------------------ */
/* Character-level tokenizer with BOS/EOS tokens                     */
/* ------------------------------------------------------------------ */

use std::collections::HashMap;

pub struct Tokenizer {
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: Vec<char>,
    #[allow(dead_code)]
    pub bos_id: usize,
    pub eos_id: usize,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut idx_to_char = vec!['<']; // BOS
        idx_to_char.push('>');           // EOS
        idx_to_char.extend(chars);

        let char_to_idx: HashMap<char, usize> = idx_to_char
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let bos_id = 0;
        let eos_id = 1;
        let vocab_size = idx_to_char.len();

        Self { char_to_idx, idx_to_char, bos_id, eos_id, vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&idx| self.idx_to_char.get(idx))
            .collect()
    }
}
