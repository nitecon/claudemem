use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexWriter, ReloadPolicy};

use crate::db::models::Memory;
use crate::error::MemoryError;

fn build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("id", STRING | STORED);
    schema_builder.add_text_field("content", TEXT | STORED);
    schema_builder.build()
}

pub fn open_or_create_index(tantivy_dir: &Path) -> Result<Index, MemoryError> {
    let schema = build_schema();

    match Index::open_in_dir(tantivy_dir) {
        Ok(index) => Ok(index),
        Err(_) => {
            // If directory exists but index is corrupted, clear it
            if tantivy_dir.exists() {
                let _ = std::fs::remove_dir_all(tantivy_dir);
                std::fs::create_dir_all(tantivy_dir)?;
            }
            Ok(Index::create_in_dir(tantivy_dir, schema)?)
        }
    }
}

pub fn index_memory(index: &Index, memory: &Memory) -> Result<(), MemoryError> {
    let schema = index.schema();
    let id_field = schema.get_field("id").unwrap();
    let content_field = schema.get_field("content").unwrap();

    let mut writer: IndexWriter = index.writer(50_000_000)?;

    // Remove existing doc with same id (for updates)
    let id_term = tantivy::Term::from_field_text(id_field, &memory.id);
    writer.delete_term(id_term);

    let mut doc = tantivy::TantivyDocument::new();
    doc.add_text(id_field, &memory.id);
    doc.add_text(content_field, &memory.content);
    writer.add_document(doc)?;
    writer.commit()?;

    Ok(())
}

pub fn remove_from_index(index: &Index, memory_id: &str) -> Result<(), MemoryError> {
    let schema = index.schema();
    let id_field = schema.get_field("id").unwrap();

    let mut writer: IndexWriter = index.writer(50_000_000)?;
    let id_term = tantivy::Term::from_field_text(id_field, memory_id);
    writer.delete_term(id_term);
    writer.commit()?;

    Ok(())
}

pub fn search_bm25(
    index: &Index,
    query_str: &str,
    limit: usize,
) -> Result<Vec<(String, f32)>, MemoryError> {
    let schema = index.schema();
    let content_field = schema.get_field("content").unwrap();
    let id_field = schema.get_field("id").unwrap();

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let query_parser = QueryParser::for_index(index, vec![content_field]);
    let query = query_parser
        .parse_query(query_str)
        .map_err(|e| MemoryError::SearchIndex(tantivy::TantivyError::InvalidArgument(e.to_string())))?;

    let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

    let mut results = Vec::new();
    for (score, doc_address) in top_docs {
        let doc: tantivy::TantivyDocument = searcher.doc(doc_address)?;
        if let Some(id_value) = doc.get_first(id_field) {
            if let Some(id_str) = id_value.as_str() {
                results.push((id_str.to_string(), score));
            }
        }
    }

    Ok(results)
}

pub fn rebuild_index(index: &Index, memories: &[Memory]) -> Result<(), MemoryError> {
    let schema = index.schema();
    let id_field = schema.get_field("id").unwrap();
    let content_field = schema.get_field("content").unwrap();

    let mut writer: IndexWriter = index.writer(50_000_000)?;
    writer.delete_all_documents()?;

    for memory in memories {
        let mut doc = tantivy::TantivyDocument::new();
        doc.add_text(id_field, &memory.id);
        doc.add_text(content_field, &memory.content);
        writer.add_document(doc)?;
    }

    writer.commit()?;
    Ok(())
}
