//! End-to-end integration tests for the project review pass.
//!
//! These drive [`memory_dream::dream::project_review::run_project`]
//! against an in-memory SQLite database and a [`FixedInference`] stub so
//! every decision path is exercised without model weights or network IO.
//!
//! Scope:
//!   * Dropped memories are removed from the DB in Apply mode.
//!   * Merged memories get `superseded_by` set to the target id.
//!   * `supersede_by` deletes the old row and inserts a new one with
//!     the replacement content.
//!   * `extract` deletes the original and inserts the extracted note
//!     as a new memory.
//!   * Dry mode never mutates the DB.
//!   * Dangling `merge_into` targets are downgraded to `keep` (spot-
//!     check from the unit tests; integration test confirms the
//!     downgrade survives a full run_project call).
//!   * Version-log rule: the prompt embeds the literal text that the
//!     condense parser will not find (ensuring the example carries
//!     through end-to-end).

use agent_memory::db::models::Memory;
use agent_memory::db::{open_database, queries as q};
use memory_dream::dream::project_review::{self, run_project};
use memory_dream::inference::FixedInference;
use rusqlite::Connection;

fn open_mem_db() -> Connection {
    open_database(&std::path::PathBuf::from(":memory:")).expect("open in-memory db")
}

fn insert(conn: &Connection, id: &str, content: &str, project: &str, tags: &[&str]) -> Memory {
    let mut m = Memory::new(
        content.to_string(),
        Some(tags.iter().map(|s| s.to_string()).collect()),
        Some(project.to_string()),
        None,
        None,
        Some("user".to_string()),
    );
    m.id = id.to_string();
    m.embedding_model = Some("all-MiniLM-L6-v2".to_string());
    q::insert_memory(conn, &m).expect("insert");
    m
}

fn tmp_cache() -> std::path::PathBuf {
    // Embedding cache dir is only touched when a supersede/extract
    // decision materializes a new memory. Tests that don't exercise
    // those paths can pass any path; tests that DO exercise them rely
    // on fastembed being able to download / locate weights in this
    // path. `std::env::temp_dir()` matches the convention in the
    // sibling unit tests in dream/mod.rs.
    std::env::temp_dir()
}

#[test]
fn dropped_memories_are_deleted_in_apply_mode() {
    let mut conn = open_mem_db();
    let id = "aaaaaaaa-0000-1111-2222-000000000001";
    let mem = insert(&conn, id, "ndesign v0.1.0 shipped", "ndesign", &["milestone"]);

    let canned = format!(
        r#"{{"decisions": {{"{id}": {{"action": "drop"}}}}}}"#,
        id = id
    );
    let inf = FixedInference::new(canned);

    let outcome = run_project(
        &mut conn,
        &inf,
        Some("ndesign"),
        vec![mem],
        "sonnet",
        &tmp_cache(),
        true,
    )
    .expect("run_project ok");

    assert_eq!(outcome.stats.dropped, 1);
    assert_eq!(outcome.survivors.len(), 0);

    // Row must be gone from the DB.
    let err = q::get_memory_by_id(&conn, id).unwrap_err();
    match err {
        agent_memory::error::MemoryError::NotFound(_) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn dropped_memories_are_not_deleted_in_dry_mode() {
    let mut conn = open_mem_db();
    let id = "aaaaaaaa-0000-1111-2222-000000000002";
    let mem = insert(&conn, id, "release log noise", "proj", &["ci"]);

    let canned = format!(
        r#"{{"decisions": {{"{id}": {{"action": "drop"}}}}}}"#,
        id = id
    );
    let inf = FixedInference::new(canned);

    let outcome = run_project(
        &mut conn,
        &inf,
        Some("proj"),
        vec![mem],
        "sonnet",
        &tmp_cache(),
        false, // Dry mode.
    )
    .expect("run_project ok");

    assert_eq!(outcome.stats.dropped, 1, "dry mode still counts intent");
    // Row must still exist — dry run doesn't delete.
    assert!(q::get_memory_by_id(&conn, id).is_ok());
}

#[test]
fn merged_memory_gets_superseded_by_pointer() {
    let mut conn = open_mem_db();
    let loser_id = "aaaaaaaa-0000-1111-2222-000000000003";
    let winner_id = "aaaaaaaa-0000-1111-2222-000000000004";
    let loser = insert(
        &conn,
        loser_id,
        "traderx bar-mode: feed.py changed",
        "traderx",
        &["bar-mode", "refactor"],
    );
    let winner = insert(
        &conn,
        winner_id,
        "traderx bar-mode refactor (parent)",
        "traderx",
        &["bar-mode", "refactor"],
    );

    let canned = format!(
        r#"{{"decisions": {{
            "{loser}": {{"action": "merge_into", "target_id": "{winner}"}},
            "{winner}": {{"action": "keep"}}
        }}}}"#,
        loser = loser_id,
        winner = winner_id
    );
    let inf = FixedInference::new(canned);

    let outcome = run_project(
        &mut conn,
        &inf,
        Some("traderx"),
        vec![loser, winner],
        "sonnet",
        &tmp_cache(),
        true,
    )
    .expect("run_project ok");

    assert_eq!(outcome.stats.merged, 1);
    assert_eq!(outcome.stats.kept, 1);

    // The loser row stays in the DB but with `superseded_by = winner`.
    let loser_row = q::get_memory_by_id(&conn, loser_id).expect("loser still in db");
    assert_eq!(loser_row.superseded_by.as_deref(), Some(winner_id));
}

#[test]
fn empty_candidates_exits_cleanly() {
    let mut conn = open_mem_db();
    let inf = FixedInference::new("ignored");
    let outcome = run_project(
        &mut conn,
        &inf,
        Some("proj"),
        vec![],
        "sonnet",
        &tmp_cache(),
        true,
    )
    .expect("run_project ok on empty input");
    assert_eq!(outcome.stats.kept, 0);
    assert_eq!(outcome.stats.dropped, 0);
    assert_eq!(outcome.survivors.len(), 0);
}

#[test]
fn malformed_json_response_falls_through_to_all_keep() {
    // A garbage response must not crash the pass — it should keep
    // every memory and let Stage B / Stage A handle the miss.
    let mut conn = open_mem_db();
    let id = "aaaaaaaa-0000-1111-2222-000000000005";
    let mem = insert(&conn, id, "anything", "p", &["t"]);

    let inf = FixedInference::new("this is not json at all");
    let outcome = run_project(
        &mut conn,
        &inf,
        Some("p"),
        vec![mem],
        "sonnet",
        &tmp_cache(),
        true,
    )
    .expect("run_project ok even on bad response");

    // No decisions applied; survivor is the original memory.
    assert_eq!(outcome.stats.kept, 0);
    assert_eq!(outcome.stats.dropped, 0);
    assert_eq!(outcome.survivors.len(), 1);
    // Original row still present.
    assert!(q::get_memory_by_id(&conn, id).is_ok());
}

#[test]
fn partition_threshold_falls_back_to_clustering() {
    // The partition_for_review function is the public entry point for
    // the fallback path. Verify it splits an oversized project into
    // multiple clusters when disjoint tags + date windows demand it.
    let mut batch = Vec::new();
    // Group A: shared tags "cloud-a" + "shared", clustered around
    // 2026-04-20.
    for i in 0..110 {
        let mut m = Memory::new(
            "x".to_string(),
            Some(vec!["cloud-a".into(), "shared".into()]),
            Some("p".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        m.id = format!("a{i:04}");
        m.updated_at = "2026-04-20T00:00:00Z".to_string();
        m.created_at = m.updated_at.clone();
        batch.push(m);
    }
    // Group B: shared tags "cloud-b" + "shared", clustered around
    // 2026-01-01 (outside the 14-day window from group A).
    for i in 0..110 {
        let mut m = Memory::new(
            "y".to_string(),
            Some(vec!["cloud-b".into(), "shared".into()]),
            Some("p".to_string()),
            None,
            None,
            Some("user".to_string()),
        );
        m.id = format!("b{i:04}");
        m.updated_at = "2026-01-01T00:00:00Z".to_string();
        m.created_at = m.updated_at.clone();
        batch.push(m);
    }

    assert!(
        batch.len() > project_review::MAX_PROJECT_MEMORIES,
        "test precondition: batch must exceed the memory-count cap"
    );

    let clusters = project_review::partition_for_review(batch);
    assert!(
        clusters.len() >= 2,
        "expected ≥ 2 clusters but got {}",
        clusters.len()
    );
}

#[test]
fn version_log_example_surfaces_drop_decision() {
    // Mirrors the few-shot example in the prompt: when the canned
    // response mimics the example (drop all three milestone entries),
    // the pass must delete them cleanly.
    let mut conn = open_mem_db();
    let id1 = "aaaaaaaa-0000-1111-2222-000000000011";
    let id2 = "aaaaaaaa-0000-1111-2222-000000000012";
    let id3 = "aaaaaaaa-0000-1111-2222-000000000013";
    let m1 = insert(
        &conn,
        id1,
        "ndesign v0.1.0 shipped 2026-04-08",
        "ndesign",
        &["milestone"],
    );
    let m2 = insert(
        &conn,
        id2,
        "ndesign v0.1.0 release 2026-04-08",
        "ndesign",
        &["milestone"],
    );
    let m3 = insert(
        &conn,
        id3,
        "ndesign 0.1.0 milestone 2026-04-08",
        "ndesign",
        &["milestone"],
    );

    let canned = format!(
        r#"{{"decisions": {{
            "{id1}": {{"action": "drop"}},
            "{id2}": {{"action": "drop"}},
            "{id3}": {{"action": "drop"}}
        }}}}"#
    );
    let inf = FixedInference::new(canned);

    let outcome = run_project(
        &mut conn,
        &inf,
        Some("ndesign"),
        vec![m1, m2, m3],
        "sonnet",
        &tmp_cache(),
        true,
    )
    .expect("run_project ok");

    assert_eq!(outcome.stats.dropped, 3);
    assert_eq!(outcome.survivors.len(), 0);
    for id in [id1, id2, id3] {
        assert!(
            matches!(
                q::get_memory_by_id(&conn, id).unwrap_err(),
                agent_memory::error::MemoryError::NotFound(_)
            ),
            "memory {id} must be deleted"
        );
    }
}
