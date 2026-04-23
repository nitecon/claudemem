//! Dream orchestrator. Wires condense + dedup + embedding + persistence
//! together into a single batch pass. The orchestrator entry point
//! ([`run`]) lives further down; this module also re-exports the
//! prompt/condense/dedup submodules for external consumers (tests +
//! the bin).

pub mod condense;
pub mod dedup;
pub mod prompt;
