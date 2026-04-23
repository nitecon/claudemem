//! Light-XML rendering helpers for CLI and MCP output.
//!
//! The project deliberately emits *light XML* rather than JSON or full XML:
//! section tags group related content, but the payload inside each section is
//! plain numbered lines. The goal is to give the consuming agent a structural
//! signal while keeping token overhead minimal — no repeated key names, no
//! attribute noise on every row.
//!
//! All functions in this module produce owned `String`s. Callers (the CLI
//! and the MCP server) are responsible for printing or returning them.
//! This module has zero DB or IO dependencies — it's a pure formatter.
//!
//! Shape reference:
//!
//! ```text
//! <project_memories>
//! 1. preview text [tag1,tag2]: (ID:4c82c482)
//! </project_memories>
//! <general_knowledge>
//! 1. preview text [tag1] (ID:372bd79d)
//! </general_knowledge>
//! <other_projects>
//! 1. colorithmic: preview text [tag1] (ID:23d0142a)
//! </other_projects>
//! <hint>Optional reflection prompt.</hint>
//! ```
//!
//! Sections are elided when empty. Mutation commands use a single self-closing
//! `<result .../>` line. `memory get` uses a `<memory>` wrapper. Ambiguous
//! short-ID resolution produces an `<ambiguous>` block with numbered candidates.

use crate::db::models::Memory;
use crate::search::SearchResult;

/// Number of leading hex characters shown for UUIDs in agent-visible output.
///
/// 8 chars gives ~4.3 billion distinct prefixes, which is far more than any
/// realistic single-user memory DB will contain, so collisions are vanishingly
/// rare. When they do happen (two memories sharing the same 8-char prefix)
/// the `<ambiguous>` path kicks in.
pub const SHORT_ID_CHARS: usize = 8;

/// Return the first `SHORT_ID_CHARS` of a UUID for display. The slice is by
/// bytes, not Unicode scalars — UUIDs are ASCII hex so this is safe. If the
/// input is shorter than `SHORT_ID_CHARS` (shouldn't happen for real UUIDs,
/// but defensive for test inputs) the whole string is returned.
pub fn short_id(id: &str) -> &str {
    if id.len() <= SHORT_ID_CHARS {
        id
    } else {
        &id[..SHORT_ID_CHARS]
    }
}

/// Render a `context` / `search` / `recall` ranked-result list as grouped
/// light-XML sections.
///
/// `current_project` is the cwd-derived (or explicitly-passed) project ident;
/// when `Some`, memories tagged with this project land in `<project_memories>`
/// and everything else (except global-scope) lands in `<other_projects>`.
/// When `None`, every non-global result lands in `<project_memories>` (the
/// distinction disappears in flat-ranking mode).
///
/// `global_sentinel` is the reserved project ident that flags a memory as
/// global-scope. Those always route to `<general_knowledge>` when present.
///
/// Sections are elided entirely when empty. A trailing `<hint>` line is
/// appended when `hint` is non-empty.
pub fn render_search_results(
    results: &[SearchResult],
    current_project: Option<&str>,
    hint: Option<&str>,
) -> String {
    let mut project_lines: Vec<String> = Vec::new();
    let mut global_lines: Vec<String> = Vec::new();
    let mut other_lines: Vec<String> = Vec::new();

    for r in results {
        // Global-scope memories always route to <general_knowledge> regardless
        // of whether a current-project was set; they're never "current".
        if r.is_global {
            global_lines.push(format_result_line(
                global_lines.len() + 1,
                &r.memory,
                /*show_project=*/ false,
            ));
        } else if r.is_current_project {
            project_lines.push(format_result_line(
                project_lines.len() + 1,
                &r.memory,
                false,
            ));
        } else if current_project.is_some() {
            // Cross-project hits during a scoped retrieval are prior-art;
            // show the originating project prefix so the agent can reason
            // about relevance.
            other_lines.push(format_result_line(
                other_lines.len() + 1,
                &r.memory,
                /*show_project=*/ true,
            ));
        } else {
            // Flat-ranking mode (no current-project) — no "other" distinction,
            // everything non-global goes in project_memories without a prefix.
            project_lines.push(format_result_line(
                project_lines.len() + 1,
                &r.memory,
                false,
            ));
        }
    }

    let mut out = String::new();
    if !project_lines.is_empty() {
        out.push_str("<project_memories>\n");
        for line in &project_lines {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("</project_memories>\n");
    }
    if !global_lines.is_empty() {
        out.push_str("<general_knowledge>\n");
        for line in &global_lines {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("</general_knowledge>\n");
    }
    if !other_lines.is_empty() {
        out.push_str("<other_projects>\n");
        for line in &other_lines {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("</other_projects>\n");
    }
    if let Some(h) = hint.filter(|s| !s.is_empty()) {
        out.push_str(&render_hint(h));
        out.push('\n');
    }
    // Trim the trailing newline so callers can `println!` without a blank line.
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Format a single ranked-result line in the shape:
///
/// ```text
/// 1. preview text [tag1,tag2] (ID:short)
/// 2. project: preview text [tag1] (ID:short)
/// ```
///
/// When `show_project` is true the memory's project ident is prepended with a
/// colon — used for the `<other_projects>` section. Tags are omitted when
/// absent rather than rendered as `[]`.
fn format_result_line(idx: usize, m: &Memory, show_project: bool) -> String {
    let preview = one_line_preview(&m.content, 160);
    let tags_part = format_tags(m.tags.as_deref());
    let id_short = short_id(&m.id);

    let project_prefix = if show_project {
        match m.project.as_deref() {
            Some(p) => format!("{p}: "),
            None => String::new(),
        }
    } else {
        String::new()
    };

    match tags_part {
        Some(tags) => format!("{idx}. {project_prefix}{preview} {tags} (ID:{id_short})"),
        None => format!("{idx}. {project_prefix}{preview} (ID:{id_short})"),
    }
}

/// Collapse multi-line content to a single line with an ellipsis cut-off so a
/// ranked-list row never spans more than one terminal line. Whitespace runs
/// (newlines, tabs, repeated spaces) are normalized to a single space.
fn one_line_preview(content: &str, max_chars: usize) -> String {
    let mut collapsed = String::with_capacity(content.len().min(max_chars * 2));
    let mut prev_space = false;
    for ch in content.chars() {
        if ch.is_whitespace() {
            if !prev_space && !collapsed.is_empty() {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
    }
    // Strip any trailing space introduced by whitespace normalization.
    while collapsed.ends_with(' ') {
        collapsed.pop();
    }

    if collapsed.chars().count() > max_chars {
        let truncated: String = collapsed.chars().take(max_chars).collect();
        format!("{truncated}…")
    } else {
        collapsed
    }
}

/// Format a tag list as `[tag1,tag2]` or `None` when empty / missing. Kept
/// bracketed to distinguish tags from other content at a glance without the
/// verbosity of an XML attribute.
fn format_tags(tags: Option<&[String]>) -> Option<String> {
    match tags {
        Some(t) if !t.is_empty() => Some(format!("[{}]", t.join(","))),
        _ => None,
    }
}

/// Render a single memory as a `<memory>` wrapper with metadata attributes
/// and the full content as the inner text. Used by `memory get`.
///
/// Attributes are emitted only when set. Content is placed on its own lines
/// so multi-line memories stay readable; XML-unsafe characters in the content
/// (`<`, `>`, `&`) are escaped so a consuming XML parser won't choke.
pub fn render_memory(m: &Memory) -> String {
    let id_short = short_id(&m.id);
    let mut attrs = format!(r#"id="{id_short}""#);
    if let Some(p) = m.project.as_deref() {
        attrs.push_str(&format!(r#" project="{}""#, escape_attr(p)));
    }
    if let Some(t) = m.memory_type.as_deref() {
        attrs.push_str(&format!(r#" type="{}""#, escape_attr(t)));
    }
    if let Some(tags) = m.tags.as_deref() {
        if !tags.is_empty() {
            attrs.push_str(&format!(r#" tags="{}""#, escape_attr(&tags.join(","))));
        }
    }
    if let Some(a) = m.agent.as_deref() {
        attrs.push_str(&format!(r#" agent="{}""#, escape_attr(a)));
    }

    format!("<memory {attrs}>\n{}\n</memory>", escape_text(&m.content))
}

/// Render a single-line mutation result, e.g.:
///
/// ```text
/// <result status="stored" id="a4936eff" scope="global"/>
/// ```
///
/// `attrs` is a slice of (key, value) pairs. `status` is passed separately to
/// guarantee ordering (status always first). Values are XML-escaped.
pub fn render_action_result(status: &str, attrs: &[(&str, String)]) -> String {
    let mut s = format!(r#"<result status="{}""#, escape_attr(status));
    for (k, v) in attrs {
        s.push_str(&format!(r#" {k}="{}""#, escape_attr(v)));
    }
    s.push_str("/>");
    s
}

/// Wrap a reflection or directive hint in a `<hint>` tag. Returns the empty
/// string when `text` is empty so callers don't have to null-check.
pub fn render_hint(text: &str) -> String {
    if text.is_empty() {
        String::new()
    } else {
        format!("<hint>{}</hint>", escape_text(text))
    }
}

/// Render an ambiguous short-ID lookup as an `<ambiguous>` block with
/// numbered candidates. Used when `resolve_id_prefix` returns `Ambiguous`.
///
/// ```text
/// <ambiguous prefix="4c82c482">
/// 1. 4c82c482-c081-4937... [colorithmic,milestone]: colorithmic v1.0.0 milestone...
/// 2. 4c82c482-d7f2-4a18... [agent-memory,schema]: Schema v3 migration design notes...
/// Reply with 1 or 2, or re-run with a longer prefix.
/// </ambiguous>
/// ```
pub fn render_ambiguous(prefix: &str, candidates: &[Memory]) -> String {
    let mut out = format!("<ambiguous prefix=\"{}\">\n", escape_attr(prefix));
    for (i, m) in candidates.iter().enumerate() {
        let idx = i + 1;
        // Show more of the UUID here (e.g. `4c82c482-c081-4937...`) so a
        // human picking visually has a clear disambiguator. Strip any
        // trailing `-` so the ellipsis doesn't look like a dangling dash.
        let id_display = if m.id.len() > 18 {
            let sliced = m.id[..18].trim_end_matches('-');
            format!("{sliced}...")
        } else {
            m.id.clone()
        };
        let preview = one_line_preview(&m.content, 80);
        let tags = format_tags(m.tags.as_deref());
        match tags {
            Some(t) => out.push_str(&format!("{idx}. {id_display} {t}: {preview}\n")),
            None => out.push_str(&format!("{idx}. {id_display}: {preview}\n")),
        }
    }
    let n = candidates.len();
    if n > 1 {
        out.push_str(&format!(
            "Reply with 1..{n}, or re-run with a longer prefix.\n"
        ));
    }
    out.push_str("</ambiguous>");
    out
}

/// Render `memory list` / `memory recall` plain text output.
///
/// One line per memory. When `memory recall` supplies a `cwd_project` we
/// annotate current-project hits with `*`. Format keeps tokens minimal and
/// leans on whitespace for readability.
pub fn render_memory_list(memories: &[Memory], cwd_project: Option<&str>) -> String {
    if memories.is_empty() {
        return "<memories count=\"0\"/>".to_string();
    }
    let mut out = format!("<memories count=\"{}\">\n", memories.len());
    for (i, m) in memories.iter().enumerate() {
        let idx = i + 1;
        let id_short = short_id(&m.id);
        let is_current = cwd_project
            .map(|cp| m.project.as_deref() == Some(cp))
            .unwrap_or(false);
        let marker = if is_current { "*" } else { " " };
        let type_label = m.memory_type.as_deref().unwrap_or("?");
        let project_label = m.project.as_deref().unwrap_or("-");
        let tags = format_tags(m.tags.as_deref()).unwrap_or_default();
        let preview = one_line_preview(&m.content, 120);
        out.push_str(&format!(
            "{idx}.{marker}({type_label}) {project_label} {tags} (ID:{id_short}): {preview}\n"
        ));
    }
    out.push_str("</memories>");
    out
}

/// Render `memory projects` output: one line per project with count, plus a
/// marker for the current-cwd project.
pub fn render_projects(rows: &[(Option<String>, i64)], cwd_project: Option<&str>) -> String {
    if rows.is_empty() {
        return "<projects count=\"0\"/>".to_string();
    }
    let mut out = format!("<projects count=\"{}\">\n", rows.len());
    for (project, count) in rows {
        let label = project.as_deref().unwrap_or("(no project)");
        let is_current = match (cwd_project, project.as_deref()) {
            (Some(cp), Some(p)) => cp == p,
            _ => false,
        };
        let marker = if is_current { "*" } else { " " };
        out.push_str(&format!("{marker}{label} ({count})\n"));
    }
    out.push_str("</projects>");
    out
}

/// Render `memory whoami` — two-line summary of derived identity. Kept out of
/// XML tags entirely since it's user-visible diagnostic output, not agent
/// context. (Currently unused at the render layer because whoami lives in the
/// `agent-tools` binary, not in this crate — kept here for symmetry if the
/// subcommand is added later.)
pub fn render_whoami(project: Option<&str>, agent_id: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str(&format!("project: {}\n", project.unwrap_or("(unknown)")));
    out.push_str(&format!("agent:   {}", agent_id.unwrap_or("(unknown)")));
    out
}

/// Escape a string for safe inclusion in an XML attribute value. `&`, `<`,
/// `>`, `"` and `'` are replaced with their named entities. Keeps the
/// produced XML parseable by a strict reader without requiring CDATA wrapping.
fn escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

/// Escape a string for safe inclusion as element text. Only `&`, `<`, and `>`
/// need replacing in text position.
fn escape_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            c => out.push(c),
        }
    }
    out
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fusion::RankedResult;
    use crate::search::MatchQuality;

    fn mk_memory(id: &str, content: &str, project: Option<&str>, tags: Option<Vec<&str>>) -> Memory {
        Memory {
            id: id.to_string(),
            content: content.to_string(),
            tags: tags.map(|ts| ts.into_iter().map(String::from).collect()),
            project: project.map(String::from),
            agent: None,
            source_file: None,
            created_at: "2026-04-23T00:00:00Z".to_string(),
            updated_at: "2026-04-23T00:00:00Z".to_string(),
            access_count: 0,
            embedding: None,
            memory_type: Some("user".to_string()),
        }
    }

    fn mk_result(memory: Memory, is_current: bool, is_global: bool) -> SearchResult {
        SearchResult {
            memory,
            rank_info: RankedResult {
                id: "rrf-id".to_string(),
                score: 0.0,
                bm25_rank: None,
                vector_rank: None,
            },
            match_quality: MatchQuality::Medium,
            is_current_project: is_current,
            is_global,
        }
    }

    #[test]
    fn short_id_slices_to_8_chars() {
        assert_eq!(short_id("a4936eff-1234-5678-9abc-def012345678"), "a4936eff");
    }

    #[test]
    fn short_id_handles_short_input() {
        assert_eq!(short_id("abc"), "abc");
    }

    #[test]
    fn render_action_result_stored() {
        let s = render_action_result(
            "stored",
            &[
                ("id", "a4936eff".to_string()),
                ("scope", "global".to_string()),
            ],
        );
        assert_eq!(s, r#"<result status="stored" id="a4936eff" scope="global"/>"#);
    }

    #[test]
    fn render_action_result_no_attrs() {
        let s = render_action_result("no_matches", &[]);
        assert_eq!(s, r#"<result status="no_matches"/>"#);
    }

    #[test]
    fn render_hint_wraps_text() {
        assert_eq!(render_hint("hello"), "<hint>hello</hint>");
    }

    #[test]
    fn render_hint_empty_returns_empty() {
        assert_eq!(render_hint(""), "");
    }

    #[test]
    fn render_memory_emits_attributes_and_content() {
        let m = mk_memory(
            "a4936eff-1234-5678-9abc-def012345678",
            "User never wants PRs opened.",
            Some("agent-memory"),
            Some(vec!["workflow", "pr"]),
        );
        let s = render_memory(&m);
        assert!(s.starts_with("<memory id=\"a4936eff\""));
        assert!(s.contains("project=\"agent-memory\""));
        assert!(s.contains("type=\"user\""));
        assert!(s.contains("tags=\"workflow,pr\""));
        assert!(s.contains("User never wants PRs opened."));
        assert!(s.ends_with("</memory>"));
    }

    #[test]
    fn render_memory_escapes_content_specials() {
        let m = mk_memory("id1", "a <b> & c", None, None);
        let s = render_memory(&m);
        assert!(s.contains("a &lt;b&gt; &amp; c"));
    }

    #[test]
    fn render_search_results_groups_sections() {
        let current = mk_memory(
            "11111111-aaaa",
            "current project memory",
            Some("agent-memory"),
            Some(vec!["local"]),
        );
        let global = mk_memory(
            "22222222-bbbb",
            "global directive",
            Some("__global__"),
            Some(vec!["pref"]),
        );
        let other = mk_memory(
            "33333333-cccc",
            "cross project prior art",
            Some("colorithmic"),
            Some(vec!["ref"]),
        );
        let results = vec![
            mk_result(current, true, false),
            mk_result(global, false, true),
            mk_result(other, false, false),
        ];
        let out = render_search_results(&results, Some("agent-memory"), None);
        assert!(out.contains("<project_memories>"));
        assert!(out.contains("current project memory [local] (ID:11111111)"));
        assert!(out.contains("<general_knowledge>"));
        assert!(out.contains("global directive [pref] (ID:22222222)"));
        assert!(out.contains("<other_projects>"));
        assert!(out.contains("colorithmic: cross project prior art [ref] (ID:33333333)"));
    }

    #[test]
    fn render_search_results_elides_empty_sections() {
        let only_global = mk_memory(
            "dddddddd-1111",
            "only global",
            Some("__global__"),
            None,
        );
        let results = vec![mk_result(only_global, false, true)];
        let out = render_search_results(&results, Some("agent-memory"), None);
        assert!(!out.contains("<project_memories>"));
        assert!(!out.contains("<other_projects>"));
        assert!(out.contains("<general_knowledge>"));
        assert!(out.contains("only global (ID:dddddddd)"));
    }

    #[test]
    fn render_search_results_appends_hint() {
        let m = mk_memory("eeeeeeee-2222", "x", Some("p"), None);
        let results = vec![mk_result(m, false, false)];
        let out = render_search_results(&results, None, Some("Check global prefs."));
        assert!(out.ends_with("<hint>Check global prefs.</hint>"));
    }

    #[test]
    fn render_search_results_empty_input_is_empty() {
        let out = render_search_results(&[], Some("p"), None);
        assert_eq!(out, "");
    }

    #[test]
    fn render_ambiguous_numbers_candidates() {
        let a = mk_memory(
            "4c82c482-c081-4937-9999-000000000001",
            "colorithmic milestone doc",
            Some("colorithmic"),
            Some(vec!["milestone"]),
        );
        let b = mk_memory(
            "4c82c482-d7f2-4a18-9999-000000000002",
            "agent-memory schema notes",
            Some("agent-memory"),
            Some(vec!["schema"]),
        );
        let out = render_ambiguous("4c82c482", &[a, b]);
        assert!(out.starts_with("<ambiguous prefix=\"4c82c482\">"));
        assert!(out.contains("1. 4c82c482-c081-4937..."));
        assert!(out.contains("2. 4c82c482-d7f2-4a18..."));
        assert!(out.contains("Reply with 1..2"));
        assert!(out.ends_with("</ambiguous>"));
    }

    #[test]
    fn one_line_preview_collapses_whitespace() {
        assert_eq!(one_line_preview("a\n\nb\tc", 99), "a b c");
    }

    #[test]
    fn one_line_preview_truncates_with_ellipsis() {
        let out = one_line_preview("abcdefghij", 5);
        assert_eq!(out, "abcde…");
    }

    #[test]
    fn render_memory_list_empty() {
        let out = render_memory_list(&[], None);
        assert_eq!(out, "<memories count=\"0\"/>");
    }

    #[test]
    fn render_memory_list_marks_current_project() {
        let a = mk_memory("aaaaaaaa-1", "local", Some("agent-memory"), None);
        let b = mk_memory("bbbbbbbb-1", "other", Some("colorithmic"), None);
        let out = render_memory_list(&[a, b], Some("agent-memory"));
        assert!(out.contains("<memories count=\"2\">"));
        assert!(out.contains("1.*(user) agent-memory"));
        assert!(out.contains("2. (user) colorithmic"));
    }

    #[test]
    fn render_projects_marks_current() {
        let rows = vec![
            (Some("agent-memory".to_string()), 42_i64),
            (Some("colorithmic".to_string()), 7),
            (None, 3),
        ];
        let out = render_projects(&rows, Some("agent-memory"));
        assert!(out.contains("*agent-memory (42)"));
        assert!(out.contains(" colorithmic (7)"));
        assert!(out.contains(" (no project) (3)"));
    }
}
