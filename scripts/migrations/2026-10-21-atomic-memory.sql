-- Additive, rerunnable migration. Old chat rows retain unknown provenance (fact=NULL).
BEGIN;
ALTER TABLE long_term_memory ADD COLUMN IF NOT EXISTS base_importance REAL;
ALTER TABLE long_term_memory ADD COLUMN IF NOT EXISTS decay_anchor TIMESTAMP;
ALTER TABLE long_term_memory ADD COLUMN IF NOT EXISTS fact JSONB;
-- Existing scores already include unknown historical decay. Anchor them at migration time.
UPDATE long_term_memory SET base_importance = importance, decay_anchor = CURRENT_TIMESTAMP AT TIME ZONE 'UTC'
WHERE base_importance IS NULL;
CREATE INDEX IF NOT EXISTS idx_ltm_active_facts ON long_term_memory
    (username, ((fact->>'status'))) WHERE fact IS NOT NULL;
COMMIT;
