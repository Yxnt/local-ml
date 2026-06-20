# Personal Memory Evolution Design

## Goal

Build the first stage of a local-first self-evolving personal agent: an
audit-first personal memory evolution foundation.

The user ultimately wants four forms of evolution, in this priority order:

1. Personal memory evolution
2. Capability/tool evolution
3. Agent behavior and strategy evolution
4. Local model fine-tuning

This design covers the first stage. It creates the trusted personal memory
substrate that later stages can safely consume.

## User-Visible Outcome

The first visible product is a responsive Web/PWA review console for daily and
weekly reflection.

The console should let the user:

- review daily and weekly summaries
- inspect candidate long-term memories
- approve, edit, reject, and revoke learned memories
- inspect why a memory exists and which evidence supports it
- distinguish locally generated summaries from remote-assisted summaries

The first version should work well on both phone and desktop. Phone use should
optimize for quick daily review. Desktop use should optimize for timeline
inspection, batch review, and memory ledger management.

## Scope

In scope:

- Apple Photos/lifelog-derived photo events and photo metadata
- Obsidian Vault ingestion as the first notes source
- Apple Health/Fitness daily summaries as the first activity/body-state source
- local model extraction and preprocessing
- remote model use only on redacted summaries
- candidate learning with explicit user confirmation
- complete audit logging for memory state changes
- reversible approved memories
- interfaces that future tool evolution, behavior evolution, and fine-tuning can
  consume

Out of scope for this stage:

- automatic high-confidence writes without review
- direct training or LoRA fine-tuning
- full Calendar ingestion
- uploading raw photos, raw notes, or detailed Health records to remote models
- broad redesign of the existing tool-evolution subsystem
- external account automation

## Existing Project Context

The repo already contains useful building blocks:

- `lifelog/` can read photos, extract vision signals, cluster events, and write
  journal drafts.
- `memory/` contains local persistence for events, preferences, feedback, and
  lightweight vector search.
- `evolution/` contains prompt and journal-style adaptation.
- `server/tools/` contains a separate tool-evolution track for generated tools,
  candidate status, verification, absorption, and metrics.
- Existing design docs already describe Apple Photos integration and in-situ
  tool evolution.

This design should not collapse those responsibilities into one large module.
Instead, it should add a dedicated orchestration layer for personal memory
evolution.

## Architecture

Create a new application boundary, tentatively named `personal_evolution/`.

Existing modules keep their current roles:

- `lifelog/`: photo-to-event and journal-draft pipeline
- `memory/`: persistence primitives for memory-like records
- `evolution/`: prompt/style/strategy adaptation
- `server/tools/`: tool capability evolution

The new `personal_evolution/` layer owns multi-source orchestration, candidate
learning, review workflow, and audit semantics.

### Components

#### Source Ingestors

Source ingestors convert source-specific data into local, low-sensitivity
summaries plus evidence references.

Initial ingestors:

- Photos/lifelog ingestor
  - uses photo timestamps, locations, and local vision summaries
  - reuses `lifelog` where possible
- Obsidian ingestor
  - indexes a configured Obsidian Vault
  - extracts note-level topics, entities, and redacted snippets
  - stores note paths and content hashes as evidence references
- Health/Fitness ingestor
  - imports daily aggregate metrics such as steps, workouts, and sleep summary
  - avoids storing or sending detailed raw Health records in remote prompts

#### Evidence Store

The evidence store persists source references and redacted summaries. It does
not copy raw photo files, raw Obsidian note bodies, or detailed Health records
into the main personal evolution database.

Evidence should be enough to answer:

- which source produced this signal?
- when was it observed?
- what low-sensitivity summary was extracted?
- how can the local system re-open the original source if the user asks?

#### Observed Event Builder

The observed event builder merges related evidence into a local timeline. It
should support facts such as:

- a photo cluster happened at a time and approximate place
- a note discussed a project, person, theme, or decision
- a day had unusual activity, rest, or movement patterns

Observed events are factual timeline objects. They are not yet long-term
personal conclusions.

#### Candidate Memory Generator

The candidate memory generator turns observed events and source summaries into
reviewable learning proposals.

Candidate types include:

- event memory
- preference
- recurring pattern
- personal insight
- health or activity correlation
- writing or reflection style preference

Local models should perform extraction and preprocessing first. Remote models
may assist with higher-level synthesis only after inputs are redacted.

#### Review Workflow

The review workflow owns state transitions:

`observed -> candidate -> approved/rejected -> revoked`

In the first stage, candidate memories default to `pending`. The user must
approve them before they become long-term memories.

Future automation can add policy-based auto-approval without changing the core
state model.

#### Audit Log

Every meaningful state change must append an audit event. Audit history must not
be overwritten.

Audited actions include:

- evidence created
- observed event created
- candidate memory created
- candidate edited
- candidate approved
- candidate rejected
- approved memory revoked
- automatic write performed by future policy
- batch migration or backfill

The audit log is what makes later automation acceptable: the system can learn
more aggressively only if the user can inspect and undo what happened.

#### Review API And PWA

The backend should expose review-oriented APIs rather than generic database
CRUD. The PWA should be a review console, not a broad dashboard.

Core API surfaces:

- daily and weekly review summary
- candidate memory queue
- approve/edit/reject candidate
- approved memory ledger
- revoke approved memory
- evidence lookup
- audit event lookup

## Data Model

### Evidence

Evidence represents a source-backed signal.

Fields should include:

- `evidence_id`
- `source_type`
- `source_ref`
- `observed_at`
- `summary`
- `sensitivity`
- `content_hash`
- `metadata`
- `created_at`

`source_ref` can point to a photo UUID, Obsidian path plus block/hash, or Health
daily aggregate identifier. It should not require duplicating raw source data.

### ObservedEvent

Observed events are factual timeline records built from evidence.

Fields should include:

- `event_id`
- `start_at`
- `end_at`
- `title`
- `summary`
- `evidence_ids`
- `confidence`
- `created_at`

### CandidateMemory

Candidate memories are learning drafts awaiting review.

Fields should include:

- `candidate_id`
- `memory_type`
- `claim`
- `rationale`
- `evidence_ids`
- `status`
- `confidence`
- `source_model`
- `remote_assisted`
- `created_at`
- `updated_at`

### ApprovedMemory

Approved memories are long-term records that can influence future summaries,
retrieval, agent context, behavior adaptation, and eventually training data
exports.

Fields should include:

- `memory_id`
- `memory_type`
- `content`
- `evidence_ids`
- `candidate_id`
- `version`
- `confidence`
- `status`
- `approved_at`
- `revoked_at`

### AuditEvent

Audit events are append-only history.

Fields should include:

- `audit_id`
- `entity_type`
- `entity_id`
- `action`
- `actor`
- `before`
- `after`
- `reason`
- `created_at`

## Privacy Boundary

The default privacy policy is:

Raw data stays local. Redacted summaries may be sent remotely.

Raw data includes:

- photo files
- full Obsidian note text
- detailed Health/Fitness records
- local file paths when not needed by the remote model
- exact sensitive locations unless explicitly allowed

Remote-assisted synthesis may receive:

- redacted event summaries
- redacted candidate claims
- coarse temporal context
- low-sensitivity aggregate metrics
- non-sensitive tags and themes

Remote-assisted outputs must be marked so the user can see which summaries or
candidates used a remote model.

## Data Flow

1. Source ingestors read Photos/lifelog data, Obsidian Vault notes, and
   Health/Fitness daily aggregates.
2. Local extractors produce low-sensitivity summaries, entities, themes, and
   metric summaries.
3. Evidence Store persists references and summaries.
4. Observed Event Builder creates a timeline from related evidence.
5. Candidate Memory Generator proposes long-term memories, preferences, and
   patterns.
6. Optional remote synthesis sees only redacted summaries.
7. PWA Review Console presents daily/weekly review and the learning queue.
8. User approval creates ApprovedMemory records.
9. Every transition appends AuditEvent records.
10. Revocation marks approved memories as revoked while preserving history.

## Web/PWA Review Console

The first PWA should include three primary surfaces.

### Review Today/Week

Shows the daily or weekly reflection:

- timeline of observed events
- photo-derived events
- Obsidian-related context
- Health/Fitness signals
- generated narrative summary
- candidate memories produced from the period

### Learning Queue

Shows pending candidate memories:

- claim
- rationale
- confidence
- source evidence
- local-only or remote-assisted marker
- approve
- edit and approve
- reject

### Memory Ledger

Shows approved and revoked memories:

- search and filter
- source evidence
- audit history
- revoke action
- status and version

## Future Evolution Hooks

### Capability/Tool Evolution

Tool evolution should consume gaps discovered by personal memory ingestion.
Examples:

- a missing importer for a source format
- a missing parser for a note convention
- a missing summarizer for an activity export

Generated tools should remain candidates until verified and promoted by the
existing tool-evolution safety path.

### Behavior And Strategy Evolution

Approved memories can later adjust:

- journal style
- summary depth
- retrieval ranking
- which candidate insights need confirmation
- agent system context

This stage should not auto-edit behavior prompts broadly. It should create the
approved memory substrate that makes those changes grounded.

### Local Model Fine-Tuning

Fine-tuning should only use exportable training examples derived from approved
and audited memories. Raw private data should not be treated as training data by
default.

The future export path should preserve:

- source memory IDs
- approval state
- revocation state
- redaction status
- intended training objective

## Error Handling

- If a source ingestor fails, record a recoverable ingestion error and continue
  with other sources.
- If local extraction fails for an evidence item, store the evidence reference
  with an extraction failure status.
- If remote synthesis is unavailable, generate local-only candidates or defer
  candidate creation.
- If approval fails, do not partially create an approved memory without an audit
  event.
- If revocation fails, preserve the original approved memory and surface the
  failure to the review API.

## Testing Strategy

### Unit Tests

- state transitions for candidates and approved memories
- audit log append-only behavior
- revocation semantics
- redaction helpers
- evidence reference serialization

### Integration Tests

- mock Photos/lifelog data into evidence and observed events
- mock Obsidian Vault notes into evidence and candidate memories
- mock Health/Fitness daily summaries into observed events
- multi-source evidence into a single review summary

### API Tests

- list review periods
- list candidate queue
- approve candidate
- edit and approve candidate
- reject candidate
- revoke approved memory
- inspect audit history

### PWA Tests

- phone and desktop responsive layouts
- long text does not overflow controls
- approve/edit/reject buttons are interactive
- revoked memories update state correctly
- remote-assisted labels render where needed

## Acceptance Criteria

The first stage is complete when:

- mock Photos/lifelog, Obsidian, and Health/Fitness inputs can produce evidence
  records
- related evidence can produce observed events
- observed events can produce pending candidate memories
- the PWA can show daily/weekly review and the learning queue
- the user can approve, edit, reject, and revoke memories
- approved memories keep source evidence and audit history
- revoked memories no longer influence future memory context
- remote-assisted candidates are clearly marked
- raw source data is not sent to remote model calls
- an end-to-end test proves:
  `evidence -> observed event -> candidate -> approval -> approved memory -> revocation -> audit lookup`

## Risks And Mitigations

### The First Version Becomes Too Broad

Mitigation: keep data source support narrow and mock-friendly. Build the common
state model first, then deepen each source.

### Candidate Memories Become Untrustworthy

Mitigation: default to pending review, require evidence, preserve audit history,
and make revocation first-class.

### Remote Summaries Leak Sensitive Data

Mitigation: enforce a redaction boundary before remote calls and mark all
remote-assisted outputs.

### The PWA Turns Into A Generic Dashboard

Mitigation: keep the first UI centered on review, approval, evidence, and audit.

### Later Fine-Tuning Uses Bad Data

Mitigation: export only approved, non-revoked, redacted training examples with
traceable provenance.
