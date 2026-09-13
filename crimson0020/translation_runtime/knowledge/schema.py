"""Knowledge store schema definitions and migrations."""

from __future__ import annotations

SCHEMA_VERSION = 1

INIT_SCHEMA_SQL = """
-- Schema metadata
CREATE TABLE IF NOT EXISTS schema_metadata (
    schema_version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL,
    description TEXT NOT NULL
);

-- Scopes table (isolation by app, package family, source, target)
CREATE TABLE IF NOT EXISTS scopes (
    scope_id TEXT PRIMARY KEY,
    app_id TEXT NOT NULL,
    package_family TEXT NOT NULL,
    source_language TEXT NOT NULL,
    target_language TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (app_id, package_family, source_language, target_language)
);

-- Lineages table
CREATE TABLE IF NOT EXISTS lineages (
    lineage_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    parent_lineage_id TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (scope_id) REFERENCES scopes(scope_id) ON DELETE CASCADE
);

-- Catalog fingerprints
CREATE TABLE IF NOT EXISTS catalog_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    lineage_id TEXT NOT NULL,
    item_count INTEGER NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (lineage_id) REFERENCES lineages(lineage_id) ON DELETE CASCADE
);

-- Translation segments
CREATE TABLE IF NOT EXISTS translation_segments (
    segment_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    lineage_id TEXT NOT NULL,
    marker TEXT NOT NULL,
    key TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_text TEXT NOT NULL,
    target_text TEXT NOT NULL,
    context_signature TEXT NOT NULL,
    hard_terms_hash TEXT NOT NULL,
    glossary_revision INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (scope_id) REFERENCES scopes(scope_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_segments_exact_marker ON translation_segments (scope_id, lineage_id, marker, key, source_hash);
CREATE INDEX IF NOT EXISTS idx_segments_contextual ON translation_segments (scope_id, lineage_id, source_hash, context_signature, hard_terms_hash);

-- Glossary terms
CREATE TABLE IF NOT EXISTS terms (
    term_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    source_term TEXT NOT NULL,
    target_term TEXT NOT NULL,
    state TEXT NOT NULL, -- user_locked, auto_locked, auto_suggested, conflict
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (scope_id) REFERENCES scopes(scope_id) ON DELETE CASCADE,
    UNIQUE (scope_id, source_term, target_term)
);

CREATE INDEX IF NOT EXISTS idx_terms_scope_source ON terms (scope_id, source_term);

-- Term evidence
CREATE TABLE IF NOT EXISTS term_evidence (
    evidence_id TEXT PRIMARY KEY,
    term_id TEXT NOT NULL,
    evidence_kind TEXT NOT NULL, -- standalone_legacy, route_consensus, user_override
    provenance TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (term_id) REFERENCES terms(term_id) ON DELETE CASCADE
);

-- Term conflicts
CREATE TABLE IF NOT EXISTS term_conflicts (
    conflict_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    source_term TEXT NOT NULL,
    competing_target TEXT NOT NULL,
    details TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (scope_id) REFERENCES scopes(scope_id) ON DELETE CASCADE
);

-- Route & model catalog (cache)
CREATE TABLE IF NOT EXISTS route_catalog (
    route_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    is_free INTEGER NOT NULL DEFAULT 1,
    context_limit INTEGER NOT NULL,
    capabilities TEXT NOT NULL,
    discovered_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (route_id, model_id)
);

-- Route health & circuit breaker
CREATE TABLE IF NOT EXISTS route_health (
    route_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'closed', -- closed, open, half_open
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL NOT NULL DEFAULT 0.0,
    last_failure_kind TEXT,
    last_status_code INTEGER,
    updated_at REAL NOT NULL,
    PRIMARY KEY (route_id, model_id)
);

-- Full text search (FTS5) for segments and terms
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    segment_id UNINDEXED,
    source_text,
    target_text,
    content='translation_segments',
    content_rowid='rowid'
);

-- Triggers to keep FTS index synced
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON translation_segments BEGIN
    INSERT INTO segments_fts(rowid, segment_id, source_text, target_text)
    VALUES (new.rowid, new.segment_id, new.source_text, new.target_text);
END;

CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON translation_segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, segment_id, source_text, target_text)
    VALUES ('delete', old.rowid, old.segment_id, old.source_text, old.target_text);
END;

CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON translation_segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, segment_id, source_text, target_text)
    VALUES ('delete', old.rowid, old.segment_id, old.source_text, old.target_text);
    INSERT INTO segments_fts(rowid, segment_id, source_text, target_text)
    VALUES (new.rowid, new.segment_id, new.source_text, new.target_text);
END;
"""
