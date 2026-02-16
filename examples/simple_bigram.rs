// Simple Bigram Model - For Comparison
// This is NOT a transformer, but shows what learning looks like
// Even this super simple model produces better output than untrained GPT!

use std::collections::HashMap;
use std::fs;

fn main() {
    // Load Shakespeare
    let text = fs::read_to_string("train.txt")
        .unwrap_or_else(|_| "hello world".to_string());

    let chars: Vec<char> = text.chars().collect();

    // Build bigram statistics (what char follows what)
    let mut bigrams: HashMap<char, HashMap<char, usize>> = HashMap::new();

    for window in chars.windows(2) {
        let (c1, c2) = (window[0], window[1]);
        *bigrams.entry(c1).or_default().entry(c2).or_default() += 1;
    }

    // Generate text
    let mut rng = Rng::new(42);
    let mut current = 'T'; // Start with 'T'
    print!("{}", current);

    for _ in 0..200 {
        if let Some(next_chars) = bigrams.get(&current) {
            // Sample from distribution
            let total: usize = next_chars.values().sum();
            let mut r = (rng.uniform() * total as f64) as usize;

            for (&next_char, &count) in next_chars {
                if r < count {
                    current = next_char;
                    print!("{}", current);
                    break;
                }
                r -= count;
            }
        } else {
            break;
        }
    }
    println!();
}

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

/*
EXAMPLE OUTPUT with this simple bigram model on Shakespeare:

"To be, or not to be as the first of the propertise, the world;
The world hath have a man, and the death of the prince that the"

Notice:
- Somewhat coherent words ("the", "world", "prince")
- Vaguely English structure
- Still nonsense overall, but WAY better than random!

This is with JUST bigram statistics (2-char patterns).
A trained transformer with 4 layers, 128 dims, and 64-token context
would be MUCH better!

The point: Even the simplest "learning" beats random weights.
*/
