CREATE TABLE github_repositories (
  id TEXT PRIMARY KEY,
  github_id TEXT NOT NULL UNIQUE CHECK(length(github_id) > 0 AND github_id NOT GLOB '*[^0-9]*' AND (github_id = '0' OR substr(github_id, 1, 1) <> '0')),
  owner TEXT NOT NULL COLLATE NOCASE,
  name TEXT NOT NULL COLLATE NOCASE,
  canonical_url TEXT NOT NULL UNIQUE,
  api_url TEXT NOT NULL UNIQUE,
  default_branch TEXT NOT NULL,
  description TEXT,
  language TEXT,
  license_spdx TEXT,
  topics_json TEXT NOT NULL,
  readme_url TEXT,
  readme_ref TEXT,
  readme_text TEXT,
  readme_sha256 TEXT CHECK(readme_sha256 IS NULL OR (length(readme_sha256)=64 AND readme_sha256 NOT GLOB '*[^0-9a-f]*')),
  created_at TEXT NOT NULL,
  UNIQUE(owner, name)
);

CREATE TABLE github_releases (
  id TEXT PRIMARY KEY,
  repository_id TEXT NOT NULL REFERENCES github_repositories(id),
  github_release_id TEXT NOT NULL UNIQUE CHECK(length(github_release_id) > 0 AND github_release_id NOT GLOB '*[^0-9]*' AND (github_release_id = '0' OR substr(github_release_id, 1, 1) <> '0')),
  tag_name TEXT NOT NULL,
  name TEXT,
  body TEXT,
  html_url TEXT NOT NULL UNIQUE,
  published_at TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(repository_id, tag_name)
);

CREATE TABLE source_observations (
  id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL CHECK(source_kind IN('github_repository','github_release','reddit_post','reddit_comment')),
  source_identity TEXT NOT NULL,
  github_repository_id TEXT REFERENCES github_repositories(id),
  github_release_id TEXT REFERENCES github_releases(id),
  reddit_post_id TEXT,
  reddit_comment_id TEXT,
  observed_at TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  raw_sha256 TEXT NOT NULL CHECK(length(raw_sha256)=64 AND raw_sha256 NOT GLOB '*[^0-9a-f]*'),
  raw_path TEXT NOT NULL,
  incomplete_results INTEGER NOT NULL DEFAULT 0 CHECK(incomplete_results IN(0,1)),
  rate_limit_remaining INTEGER,
  created_at TEXT NOT NULL,
  CHECK((github_repository_id IS NOT NULL) + (github_release_id IS NOT NULL) + (reddit_post_id IS NOT NULL) + (reddit_comment_id IS NOT NULL) = 1),
  CHECK(
    (source_kind='github_repository' AND github_repository_id IS NOT NULL AND github_release_id IS NULL AND reddit_post_id IS NULL AND reddit_comment_id IS NULL AND source_identity='github_repository:' || github_repository_id)
    OR (source_kind='github_release' AND github_repository_id IS NULL AND github_release_id IS NOT NULL AND reddit_post_id IS NULL AND reddit_comment_id IS NULL AND source_identity='github_release:' || github_release_id)
    OR (source_kind='reddit_post' AND github_repository_id IS NULL AND github_release_id IS NULL AND reddit_post_id IS NOT NULL AND reddit_comment_id IS NULL AND source_identity='reddit_post:' || reddit_post_id)
    OR (source_kind='reddit_comment' AND github_repository_id IS NULL AND github_release_id IS NULL AND reddit_post_id IS NULL AND reddit_comment_id IS NOT NULL AND source_identity='reddit_comment:' || reddit_comment_id)
  ),
  UNIQUE(source_identity, observed_at, raw_sha256)
);

CREATE INDEX github_repository_observations_idx ON source_observations(github_repository_id, observed_at);
CREATE INDEX github_release_observations_idx ON source_observations(github_release_id, observed_at);
