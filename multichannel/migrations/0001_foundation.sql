CREATE TABLE schema_migrations (
  migration_id TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE
    CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE channels (
  id TEXT PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  locale TEXT NOT NULL CHECK(locale='en'),
  enabled INTEGER NOT NULL CHECK(enabled IN(0,1)),
  created_at TEXT NOT NULL
);

CREATE TABLE platform_accounts (
  id TEXT PRIMARY KEY,
  channel_id TEXT NOT NULL REFERENCES channels(id),
  platform TEXT NOT NULL CHECK(platform IN('youtube','tiktok','instagram','facebook')),
  external_account_id TEXT,
  display_name TEXT,
  capability_state TEXT NOT NULL CHECK(capability_state IN('disabled','credentials_blocked','app_review_blocked','ready_manual_finish','ready_direct')),
  capability_reason TEXT,
  capability_checked_at TEXT,
  media_capability TEXT NOT NULL DEFAULT 'none' CHECK(media_capability IN('none','public_url','resumable')),
  enabled INTEGER NOT NULL CHECK(enabled IN(0,1)),
  created_at TEXT NOT NULL,
  UNIQUE(channel_id,platform),
  UNIQUE(platform,external_account_id)
);

CREATE TABLE job_runs (
  id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  subject_type TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN('queued','claimed','running','succeeded','failed','deferred','ambiguous','cancelled')),
  attempt_count INTEGER NOT NULL DEFAULT 0,
  worker_id TEXT,
  claim_token TEXT,
  claimed_at TEXT,
  started_at TEXT,
  finished_at TEXT,
  next_eligible_at TEXT,
  error_class TEXT,
  error_detail TEXT,
  partial_state_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE job_events (
  id TEXT PRIMARY KEY,
  job_run_id TEXT NOT NULL REFERENCES job_runs(id),
  event_type TEXT NOT NULL,
  from_state TEXT,
  to_state TEXT,
  actor TEXT NOT NULL,
  detail_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE requeue_requests (
  id TEXT PRIMARY KEY,
  job_run_id TEXT NOT NULL REFERENCES job_runs(id),
  request_key TEXT NOT NULL UNIQUE,
  operator TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL
);
