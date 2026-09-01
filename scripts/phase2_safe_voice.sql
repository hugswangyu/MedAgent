-- Phase 2: sentence-level safety audit fields associated by (session_id, turn_id).
ALTER TABLE voice_turns ADD COLUMN IF NOT EXISTS raw_model_text TEXT;
ALTER TABLE voice_turns ADD COLUMN IF NOT EXISTS final_tts_text TEXT;

CREATE INDEX IF NOT EXISTS idx_voice_turns_turn_id ON voice_turns(turn_id);
