//! `memory-dream` — offline batch compactor for the agent-memory database.
//!
//! One-shot CLI. Loads the configured gemma3 model via candle, walks the
//! SQLite DB that `memory` also uses, condenses verbose rows, deduplicates
//! near-identical rows via cosine similarity, and exits. Never runs as a
//! daemon, never opens a network port, never listens for anything.
//!
//! The real logic lives in the sibling library crate (`src/lib.rs`); this
//! binary just parses CLI args and dispatches.

fn main() -> anyhow::Result<()> {
    // Placeholder entry point — the real dispatch lands in a follow-up
    // commit once the submodules are wired up.
    println!("memory-dream placeholder");
    Ok(())
}
