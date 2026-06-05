"""Tests for PromptStore.

Covers:
  1. save() creates candidate
  2. promote() changes status to active
  3. get_active() returns the promoted version
  4. rollback() reverts to previous
  5. reject() marks as rejected
"""

from __future__ import annotations

import pytest

from server.optimization.prompt_store import PromptStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    db_path = str(tmp_path / "prompts.db")
    s = PromptStore(db_path=db_path)
    s.connect()
    yield s
    s.disconnect()


# ---------------------------------------------------------------------------
# save() tests
# ---------------------------------------------------------------------------


class TestSave:
    """Tests for PromptStore.save()."""

    def test_save_creates_candidate(self, store):
        """save() creates a new version with status='candidate'."""
        pv = store.save("test_prompt", "Hello world", optimizer="manual", score=0.8)
        assert pv.prompt_name == "test_prompt"
        assert pv.version == 1
        assert pv.content == "Hello world"
        assert pv.optimizer == "manual"
        assert pv.score == 0.8
        assert pv.status == "candidate"

    def test_save_auto_increments_version(self, store):
        """Multiple saves auto-increment the version number."""
        v1 = store.save("test_prompt", "Version 1", optimizer="manual")
        v2 = store.save("test_prompt", "Version 2", optimizer="manual")
        v3 = store.save("test_prompt", "Version 3", optimizer="manual")

        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3

    def test_save_different_prompts_independent(self, store):
        """Different prompt names have independent version counters."""
        a1 = store.save("prompt_a", "A v1", optimizer="manual")
        b1 = store.save("prompt_b", "B v1", optimizer="manual")
        a2 = store.save("prompt_a", "A v2", optimizer="manual")

        assert a1.version == 1
        assert b1.version == 1
        assert a2.version == 2

    def test_save_with_score_and_eval(self, store):
        """Score and eval_summary are persisted."""
        pv = store.save(
            "test_prompt",
            "Optimized prompt",
            optimizer="gepa",
            score=0.95,
            eval_summary="BootstrapFewShot, val=0.95",
        )
        assert pv.score == 0.95
        assert pv.eval_summary == "BootstrapFewShot, val=0.95"

    def test_save_with_no_score(self, store):
        """Score can be None."""
        pv = store.save("test_prompt", "content", optimizer="manual")
        assert pv.score is None

    def test_list_versions(self, store):
        """list_versions returns all versions, newest first."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.save("test_prompt", "v2", optimizer="manual")
        store.save("test_prompt", "v3", optimizer="manual")

        versions = store.list_versions("test_prompt")
        assert len(versions) == 3
        assert versions[0].version == 3  # newest first
        assert versions[1].version == 2
        assert versions[2].version == 1


# ---------------------------------------------------------------------------
# promote() tests
# ---------------------------------------------------------------------------


class TestPromote:
    """Tests for PromptStore.promote()."""

    def test_promote_candidate_to_active(self, store):
        """promote() changes a candidate to active."""
        pv = store.save("test_prompt", "v1 content", optimizer="manual")
        assert pv.status == "candidate"

        success = store.promote("test_prompt", 1)
        assert success is True

        active = store.get_active("test_prompt")
        assert active is not None
        assert active.version == 1
        assert active.status == "active"

    def test_promote_nonexistent_version(self, store):
        """promote() returns False for non-existent version."""
        success = store.promote("test_prompt", 99)
        assert success is False

    def test_promote_archives_previous_active(self, store):
        """promote() archives the previously active version."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.save("test_prompt", "v2", optimizer="manual")

        store.promote("test_prompt", 1)
        v1_active = store.get_active("test_prompt")
        assert v1_active.version == 1

        store.promote("test_prompt", 2)
        v2_active = store.get_active("test_prompt")
        assert v2_active.version == 2

        # v1 should be archived
        v1 = store.get_version("test_prompt", 1)
        assert v1.status == "archived"

    def test_promote_only_candidates(self, store):
        """promote() rejects non-candidate versions."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.promote("test_prompt", 1)  # now active

        # Try to promote the active version again
        success = store.promote("test_prompt", 1)
        assert success is False


# ---------------------------------------------------------------------------
# get_active() tests
# ---------------------------------------------------------------------------


class TestGetActive:
    """Tests for PromptStore.get_active()."""

    def test_get_active_returns_none_when_none(self, store):
        """get_active() returns None when no active version exists."""
        result = store.get_active("test_prompt")
        assert result is None

    def test_get_active_returns_promoted_version(self, store):
        """get_active() returns the promoted version."""
        store.save("test_prompt", "v1 content", optimizer="manual")
        store.save("test_prompt", "v2 content", optimizer="manual")
        store.promote("test_prompt", 2)

        active = store.get_active("test_prompt")
        assert active is not None
        assert active.version == 2
        assert active.content == "v2 content"

    def test_get_active_after_multiple_promotes(self, store):
        """get_active() returns the latest promoted version."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.save("test_prompt", "v2", optimizer="manual")
        store.save("test_prompt", "v3", optimizer="manual")

        store.promote("test_prompt", 1)
        store.promote("test_prompt", 3)

        active = store.get_active("test_prompt")
        assert active is not None
        assert active.version == 3


# ---------------------------------------------------------------------------
# rollback() tests
# ---------------------------------------------------------------------------


class TestRollback:
    """Tests for PromptStore.rollback()."""

    def test_rollback_to_previous(self, store):
        """rollback() reverts to the previously active (archived) version."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.save("test_prompt", "v2", optimizer="gepa")

        store.promote("test_prompt", 1)
        store.promote("test_prompt", 2)

        # Current active is v2
        assert store.get_active("test_prompt").version == 2

        # Rollback -> should restore v1
        success = store.rollback("test_prompt")
        assert success is True

        active = store.get_active("test_prompt")
        assert active is not None
        assert active.version == 1

    def test_rollback_marks_current_as_rejected(self, store):
        """rollback() marks the current active version as rejected."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.save("test_prompt", "v2", optimizer="gepa")

        store.promote("test_prompt", 1)
        store.promote("test_prompt", 2)
        store.rollback("test_prompt")

        v2 = store.get_version("test_prompt", 2)
        assert v2.status == "rejected"

    def test_rollback_no_archived_version(self, store):
        """rollback() returns False when no archived version exists."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.promote("test_prompt", 1)

        # No archived version -> rollback fails
        success = store.rollback("test_prompt")
        assert success is False

    def test_rollback_no_active_no_archived(self, store):
        """rollback() returns False when there's nothing to roll back to."""
        store.save("test_prompt", "v1", optimizer="manual")
        # v1 is still candidate, no active, no archived

        success = store.rollback("test_prompt")
        assert success is False


# ---------------------------------------------------------------------------
# reject() tests
# ---------------------------------------------------------------------------


class TestReject:
    """Tests for PromptStore.reject()."""

    def test_reject_candidate(self, store):
        """reject() marks a candidate version as rejected."""
        store.save("test_prompt", "v1", optimizer="manual")
        success = store.reject("test_prompt", 1)
        assert success is True

        v1 = store.get_version("test_prompt", 1)
        assert v1.status == "rejected"

    def test_reject_nonexistent(self, store):
        """reject() returns False for non-existent version."""
        success = store.reject("test_prompt", 99)
        assert success is False

    def test_reject_active_version(self, store):
        """reject() can also mark an active version as rejected."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.promote("test_prompt", 1)

        success = store.reject("test_prompt", 1)
        assert success is True

        v1 = store.get_version("test_prompt", 1)
        assert v1.status == "rejected"

    def test_get_active_after_reject(self, store):
        """get_active() returns None after the active version is rejected."""
        store.save("test_prompt", "v1", optimizer="manual")
        store.promote("test_prompt", 1)
        store.reject("test_prompt", 1)

        active = store.get_active("test_prompt")
        assert active is None
