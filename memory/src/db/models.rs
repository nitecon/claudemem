use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub access_count: i64,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,

    // -- Schema v3 (Release 2) additions -------------------------------------
    //
    // Pre-dream rows have all four of these as None; `memory-dream` populates
    // them on its first pass. Default read paths filter `superseded_by IS NULL`
    // so superseded rows stay in the DB for audit but never surface in
    // search / context / list output.
    /// Original verbatim text preserved when dream condenses `content`. When
    /// populated, `content` holds the short form and `content_raw` holds the
    /// user's original text so nothing is lost across a dream pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_raw: Option<String>,
    /// UUID of the newer memory that subsumes this one (dedup pointer). Set by
    /// the dream pass when cosine similarity (or exact match) flags this row
    /// as obsoleted by another.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
    /// Stamp identifying the prompt + model combo that produced the current
    /// `content`. Lets a future dream pass detect stale condensations and
    /// re-run them if the prompt or model has been revised.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condenser_version: Option<String>,
    /// Name of the embedder used to compute `embedding`. Dream uses this to
    /// ensure it only dedups rows that share a vector space.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
}

impl Memory {
    pub fn new(
        content: String,
        tags: Option<Vec<String>>,
        project: Option<String>,
        agent: Option<String>,
        source_file: Option<String>,
        memory_type: Option<String>,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            tags,
            project,
            agent,
            source_file,
            created_at: now.clone(),
            updated_at: now,
            access_count: 0,
            embedding: None,
            memory_type,
            content_raw: None,
            superseded_by: None,
            condenser_version: None,
            embedding_model: None,
        }
    }
}

pub fn embedding_to_blob(emb: &[f32]) -> Vec<u8> {
    emb.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
