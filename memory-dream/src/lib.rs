//! `memory-dream` library surface.
//!
//! The dream compactor is primarily a CLI (`src/main.rs`), but every unit of
//! real logic lives in this library so unit tests can exercise them without
//! invoking the binary. The bin itself is a thin entry point that parses
//! clap args and dispatches into [`dream::run`].

pub mod cli;
pub mod dream;
pub mod inference;
pub mod model_manager;
