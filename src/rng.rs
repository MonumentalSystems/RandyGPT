/* ------------------------------------------------------------------ */
/* Minimal xorshift PRNG                                             */
/* ------------------------------------------------------------------ */

pub struct Rng {
    pub state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self { Self { state: seed } }

    pub fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    pub fn gauss(&mut self, mean: f32, std: f32) -> f32 {
        let mut u1 = self.uniform();
        let u2 = self.uniform();
        if u1 < 1e-30 { u1 = 1e-30; }
        let mag = ((-2.0 * u1.ln()).sqrt()) as f32;
        mean + std * mag * ((2.0 * std::f64::consts::PI * u2).cos() as f32)
    }

    pub fn choice(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize
    }
}
