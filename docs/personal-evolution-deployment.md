# Personal Evolution Deployment Runbook

This runbook deploys and verifies the review-first personal memory evolution
flow on a Mac mini. It keeps raw personal data local, creates reviewable
candidate memories, and only injects user-approved memories into the local
agent context.

## What This Enables

- Ingest Obsidian markdown notes and local Photos/lifelog summaries into
  `personal_evolution`.
- Generate pending candidate memories for user review.
- Approve, edit, reject, and revoke memories through `/personal-evolution/*`.
- Include approved, non-revoked memories in the local model system prompt.

It does not automatically approve memories, upload raw notes/photos, or fine
tune a model.

## Prerequisites

- macOS with the target Photos library available to the service user.
- Python environment for this repo.
- Obsidian vault path known on the Mac mini.
- If using direct Photos metadata sync, install `osxphotos` in the same Python
  environment:

```bash
python -m pip install osxphotos
```

The lifelog reader can also use AppleScript/PyObjC fallbacks. Photos access may
require a one-time macOS permission prompt.

## Configure Sources

Edit `config.yaml` on the Mac mini:

```yaml
integrations:
  obsidian:
    vaults:
      main: /Users/YOUR_USER/Documents/Obsidian/Main

  photos:
    enabled: true
    photos_library: /Users/YOUR_USER/Pictures/Photos Library.photoslibrary
    db_path: memory/photos.db
```

Use the real user home path. Do not run the service as a different macOS user
unless that user can read the Photos library and Obsidian vault.

## Diagnose The Mac mini

Run this first:

```bash
python -m personal_evolution.ingest_cli --diagnose-sources
```

Expected healthy shape:

```text
obsidian:main exists=True path=/Users/YOUR_USER/Documents/Obsidian/Main
photos_library exists=True path=/Users/YOUR_USER/Pictures/Photos Library.photoslibrary
module:osxphotos available=True
module:Photos available=True
module:objc available=True
osascript available=True
```

`module:Photos` may be false if PyObjC Photos bindings are unavailable; the
Photos path can still work through other lifelog fallbacks, but verify with a
real ingest run.

## Smoke Test Without Photos

Use this when checking the deployment path before granting Photos access:

```bash
python -m personal_evolution.ingest_cli \
  --db memory/personal_evolution.sqlite3 \
  --no-photos
```

Expected output should include nonzero evidence/candidates if the Obsidian vault
contains markdown notes:

```text
personal ingestion complete: evidence=... events=... candidates=...
```

Warnings are recoverable source issues. Fix missing vault paths before relying
on the data.

## Run Real Ingestion

For today's Photos/lifelog plus configured Obsidian vaults:

```bash
python -m personal_evolution.ingest_cli \
  --db memory/personal_evolution.sqlite3
```

For a specific Photos day:

```bash
python -m personal_evolution.ingest_cli \
  --db memory/personal_evolution.sqlite3 \
  --date 2026-06-20
```

The command writes evidence, observed events, and pending candidates. It does
not approve anything automatically.

## Start The Server

Point the server and agent at the same personal evolution database:

```bash
export PERSONAL_EVOLUTION_DB="$PWD/memory/personal_evolution.sqlite3"
python -m server.main
```

Open the review console:

```text
http://localhost:8000/personal-evolution/app
```

Useful API checks:

```bash
curl http://localhost:8000/personal-evolution/candidates
curl http://localhost:8000/personal-evolution/memories
curl "http://localhost:8000/personal-evolution/audit"
```

After you approve a candidate, the agent loads approved, non-revoked memories
from `PERSONAL_EVOLUTION_DB` into its system prompt.

## Current-System Verification

These commands were used to verify the code path in this worktree:

```bash
python -m personal_evolution.ingest_cli --diagnose-sources
python -m personal_evolution.ingest_cli --db /tmp/personal.sqlite3 --no-photos
python -m pytest -q
```

The local development machine currently reports:

```text
obsidian:main exists=False
photos_library exists=True
module:osxphotos available=False
module:Photos available=False
module:objc available=True
osascript available=True
```

That means the code path is verified, but the current machine still needs the
real Obsidian vault path and Photos dependencies before real-source ingestion is
complete.

## Troubleshooting

- `obsidian:main exists=False`: update `config.yaml` or pass
  `--obsidian-vault /path/to/vault`.
- `module:osxphotos available=False`: install `osxphotos` in the active Python
  environment.
- Photos ingest returns zero items: confirm the target date has photos and that
  the service user has Photos permission.
- Candidates exist but the model does not use them: confirm
  `PERSONAL_EVOLUTION_DB` points to the same DB used by ingestion and review.
- Revoked memories still appear in API history by design, but they are excluded
  from agent context.
