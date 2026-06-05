"""ToolAbsorber — find, cluster, and merge similar tools.

Designed to run as a nightly batch job.  Three stages:

1. **Discover**: find similar tool pairs via sqlite-vec embeddings.
2. **Aggregate**: ask the LLM whether each cluster is truly mergeable.
3. **Merge**: generate a canonical merged tool, test it, register it.

Only ``candidate`` and ``active`` tools are considered.  Deprecated/archived
tools are ignored.  After a successful merge, the parent tools are deprecated
and a ``tool_lineage`` record links child → parents.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from server.optimization.prompt_store import PromptStore
from server.tools.registry import ToolRegistry
from server.tools.spec import (
    RiskLevel,
    ToolContext,
    ToolResult,
    ToolRuntime,
    ToolSpec,
    ToolStatus,
)
from server.tools.telemetry import TelemetryService
from server.tools.verifier import ToolVerifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.86  # cosine distance; lower = more similar
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 5


# ---------------------------------------------------------------------------
# Lineage table DDL
# ---------------------------------------------------------------------------

_LINEAGE_DDL = """
CREATE TABLE IF NOT EXISTS tool_lineage (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    child_tool_name     TEXT NOT NULL,
    child_tool_version  TEXT NOT NULL,
    parent_tool_name    TEXT NOT NULL,
    parent_tool_version TEXT NOT NULL,
    relation            TEXT NOT NULL,
    created_at          TEXT NOT NULL
)
"""


# ---------------------------------------------------------------------------
# Absorber
# ---------------------------------------------------------------------------


class ToolAbsorber:
    """Find and merge similar tools.

    Args:
        registry: The ToolRegistry to operate on.
        telemetry: TelemetryService for event logging.
        verifier: ToolVerifier for testing merged tools.
        remote_generate: async callable ``(system_prompt, user_prompt) -> str``
            that calls the remote LLM.  Falls back to no-op if not provided.
        sandbox_dir: Directory for generated tool files.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        telemetry: TelemetryService,
        verifier: ToolVerifier,
        remote_generate: Any = None,
        sandbox_dir: str = "server/tools/sandbox",
        prompt_store: PromptStore | None = None,
    ) -> None:
        self._registry = registry
        self._telemetry = telemetry
        self._verifier = verifier
        self._remote_generate = remote_generate
        self._sandbox_dir = sandbox_dir
        self._prompt_store = prompt_store
        self._init_lineage_table()

    def _init_lineage_table(self) -> None:
        try:
            conn = self._registry._conn
            if conn is not None:
                conn.execute(_LINEAGE_DDL)
                conn.commit()
        except Exception:
            pass

    # -- public API ----------------------------------------------------------

    async def run(self, dry_run: bool = False) -> dict[str, Any]:
        """Execute the full absorption pipeline.

        Returns::

            {
                "clusters_found": int,
                "clusters_merged": int,
                "tools_deprecated": int,
                "details": [...]
            }
        """
        stats: dict[str, Any] = {
            "clusters_found": 0,
            "clusters_merged": 0,
            "tools_deprecated": 0,
            "details": [],
        }

        # 1. Discover similar tool pairs
        pairs = self._discover_similar_pairs()
        if not pairs:
            logger.info("Absorber: no similar tool pairs found")
            return stats

        # 2. Cluster pairs into groups
        clusters = self._cluster_pairs(pairs)
        stats["clusters_found"] = len(clusters)
        logger.info("Absorber: found %d clusters", len(clusters))

        # 3. For each cluster, aggregate + merge
        for cluster in clusters:
            cluster_result = await self._process_cluster(cluster, dry_run=dry_run)
            stats["details"].append(cluster_result)

            if cluster_result.get("merged"):
                stats["clusters_merged"] += 1
                stats["tools_deprecated"] += len(cluster_result.get("deprecated", []))

        return stats

    def get_lineage(self, tool_name: str) -> list[dict[str, Any]]:
        """Return the lineage of a tool (parents it was merged from)."""
        conn = self._registry._conn
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT * FROM tool_lineage WHERE child_tool_name = ? ORDER BY created_at",
            (tool_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Stage 1: Discover ---------------------------------------------------

    def _discover_similar_pairs(self) -> list[tuple[ToolSpec, ToolSpec, float]]:
        """Find pairs of tools with embedding similarity > threshold.

        Uses sqlite-vec if available, otherwise brute-force cosine similarity.
        """
        tools = self._registry.list_tools(status=None)
        active = [t for t in tools if t.status in (ToolStatus.ACTIVE, ToolStatus.CANDIDATE)]
        if len(active) < 2:
            return []

        # Try sqlite-vec first
        pairs = self._discover_via_vec(active)
        if pairs is not None:
            return pairs

        # Fallback: brute-force cosine similarity
        return self._discover_bruteforce(active)

    def _discover_via_vec(
        self, active: list[ToolSpec]
    ) -> list[tuple[ToolSpec, ToolSpec, float]] | None:
        """Try sqlite-vec based discovery.  Returns None if unavailable."""
        # Quick probe: if search returns nothing for the first tool with embedding,
        # vec table is likely empty or unavailable.
        probe = next((t for t in active if t.embedding is not None), None)
        if probe is None:
            return None

        probe_hits = self._registry.search_by_embedding(probe.embedding, limit=5)  # type: ignore[arg-type]
        if not probe_hits:
            return None  # vec table empty or unavailable

        # Vec is available — do full discovery
        pairs: list[tuple[ToolSpec, ToolSpec, float]] = []
        seen: set[tuple[str, str]] = set()

        for tool in active:
            if tool.embedding is None:
                continue

            hits = self._registry.search_by_embedding(tool.embedding, limit=5)
            for other, dist in hits:
                if other.name == tool.name:
                    continue
                if other.status not in (ToolStatus.ACTIVE, ToolStatus.CANDIDATE):
                    continue

                pair_key = tuple(sorted([tool.name, other.name]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                if dist < (1.0 - SIMILARITY_THRESHOLD):
                    pairs.append((tool, other, dist))

        pairs.sort(key=lambda x: x[2])
        logger.info("Absorber (vec): discovered %d similar pairs", len(pairs))
        return pairs

    def _discover_bruteforce(
        self, active: list[ToolSpec]
    ) -> list[tuple[ToolSpec, ToolSpec, float]]:
        """Brute-force cosine similarity over all tool pairs."""
        import math

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        with_emb = [t for t in active if t.embedding is not None]
        pairs: list[tuple[ToolSpec, ToolSpec, float]] = []

        for i, t1 in enumerate(with_emb):
            for t2 in with_emb[i + 1:]:
                sim = cosine_sim(t1.embedding, t2.embedding)  # type: ignore[arg-type]
                if sim >= SIMILARITY_THRESHOLD:
                    # Store as distance for consistency (lower = more similar)
                    pairs.append((t1, t2, 1.0 - sim))

        pairs.sort(key=lambda x: x[2])
        logger.info("Absorber (brute-force): discovered %d similar pairs", len(pairs))
        return pairs

    # -- Stage 2: Cluster ----------------------------------------------------

    def _cluster_pairs(
        self, pairs: list[tuple[ToolSpec, ToolSpec, float]]
    ) -> list[list[ToolSpec]]:
        """Group overlapping pairs into clusters using union-find."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for t1, t2, _ in pairs:
            union(t1.name, t2.name)

        groups: dict[str, list[ToolSpec]] = {}
        tool_map: dict[str, ToolSpec] = {}
        for t1, t2, _ in pairs:
            tool_map[t1.name] = t1
            tool_map[t2.name] = t2

        for name in tool_map:
            root = find(name)
            groups.setdefault(root, [])
            if tool_map[name] not in groups[root]:
                groups[root].append(tool_map[name])

        # Filter clusters by size
        clusters = [c for c in groups.values() if MIN_CLUSTER_SIZE <= len(c) <= MAX_CLUSTER_SIZE]
        return clusters

    # -- Stage 3: Aggregate + Merge ------------------------------------------

    async def _process_cluster(
        self, cluster: list[ToolSpec], dry_run: bool = False
    ) -> dict[str, Any]:
        """Process a single cluster: aggregate → merge → verify → register."""
        names = [t.name for t in cluster]
        result: dict[str, Any] = {"cluster": names, "merged": False, "deprecated": []}

        # Aggregate: ask LLM if merge is appropriate
        should_merge, merge_reason = await self._aggregate(cluster)
        result["aggregate_reason"] = merge_reason

        if not should_merge:
            result["action"] = "skipped"
            return result

        if dry_run:
            result["action"] = "dry_run_would_merge"
            return result

        # Merge: generate canonical tool
        merge_result = await self._merge(cluster)
        if not merge_result["success"]:
            result["action"] = "merge_failed"
            result["error"] = merge_result.get("error", "")
            return result

        source_code = merge_result["source_code"]
        merged_name = merge_result["merged_name"]

        # Verify
        verify_result = self._verifier.verify(merged_name, source_code)
        if not verify_result.passed:
            result["action"] = "verify_failed"
            result["errors"] = verify_result.errors
            return result

        # Test with all parent tools' test cases
        test_passed = await self._run_parent_tests(cluster, merged_name, source_code)
        if not test_passed:
            result["action"] = "parent_tests_failed"
            return result

        # Register merged tool
        merged_spec = ToolSpec(
            name=merged_name,
            description=merge_result["description"],
            input_schema=merge_result["input_schema"],
            output_schema=merge_result["output_schema"],
            runtime=ToolRuntime.PYTHON_GENERATED,
            provider="merged",
            risk_level=max((t.risk_level for t in cluster), key=lambda r: list(RiskLevel).index(r)),
            status=ToolStatus.ACTIVE,
            metadata={
                "merged_from": names,
                "merge_reason": merge_reason,
                "source_file": merge_result.get("file_path", ""),
            },
        )
        self._registry.register(merged_spec)

        # Record creation telemetry for EGL tracking
        self._telemetry.record_tool_created(
            merged_name,
            merged_spec.version,
            metadata={
                "runtime": "python_generated",
                "provider": "merged",
                "source": "absorber",
                "risk_level": merged_spec.risk_level.value,
            },
        )

        # Deprecate parents
        for parent in cluster:
            self._registry.unregister(parent.name)
            result["deprecated"].append(parent.name)

        # Record lineage
        self._record_lineage(merged_spec, cluster, "merged_from")

        # Telemetry
        self._telemetry.record("tool_merged", tool_name=merged_name, metadata={"parents": names})

        result["merged"] = True
        result["merged_tool"] = merged_name
        result["action"] = "merged"
        logger.info("Absorber: merged %s → %s", names, merged_name)

        return result

    # -- Aggregator ----------------------------------------------------------

    async def _aggregate(self, cluster: list[ToolSpec]) -> tuple[bool, str]:
        """Ask the LLM whether this cluster should be merged.

        Returns (should_merge, reason).
        """
        if self._remote_generate is None:
            # Without LLM, merge only if names are very similar
            return self._heuristic_aggregate(cluster)

        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}\n  Input: {json.dumps(t.input_schema, ensure_ascii=False)[:200]}\n  Output: {json.dumps(t.output_schema or {}, ensure_ascii=False)[:200]}"
            for t in cluster
        )

        prompt = AGGREGATOR_PROMPT.format(tools=tool_descriptions)

        aggregator_system = AGGREGATOR_SYSTEM
        if self._prompt_store:
            try:
                active = self._prompt_store.get_active("absorber_aggregator_prompt")
                if active:
                    aggregator_system = active.content
            except Exception:
                logger.warning("Failed to read active aggregator prompt", exc_info=True)

        try:
            response = await self._remote_generate(aggregator_system, prompt)
            decision = self._parse_decision(response)
            return decision
        except Exception as e:
            logger.warning("Aggregator LLM call failed: %s", e)
            return self._heuristic_aggregate(cluster)

    def _heuristic_aggregate(self, cluster: list[ToolSpec]) -> tuple[bool, str]:
        """Fallback: merge if tools share the same provider and runtime."""
        providers = {t.provider for t in cluster}
        runtimes = {t.runtime for t in cluster}
        if len(providers) == 1 and len(runtimes) == 1:
            return True, f"Heuristic: same provider ({providers.pop()}) and runtime"
        return False, "Heuristic: different providers or runtimes"

    @staticmethod
    def _parse_decision(response: str) -> tuple[bool, str]:
        """Parse the aggregator's JSON response."""
        try:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                should_merge = data.get("merge", False)
                reason = data.get("reason", "")
                return bool(should_merge), reason
        except Exception:
            pass

        # Fallback: check for keywords
        lower = response.lower()
        if '"merge": true' in lower or '"merge":true' in lower:
            return True, response[:200]
        if '"merge": false' in lower or '"merge":false' in lower:
            return False, response[:200]

        return False, f"Could not parse aggregator response: {response[:200]}"

    # -- Merger --------------------------------------------------------------

    async def _merge(self, cluster: list[ToolSpec]) -> dict[str, Any]:
        """Generate a merged canonical tool using the LLM.

        Returns::

            {"success": bool, "merged_name": str, "source_code": str, ...}
        """
        if self._remote_generate is None:
            return self._heuristic_merge(cluster)

        tool_specs = "\n\n".join(
            f"## Tool: {t.name}\n"
            f"Description: {t.description}\n"
            f"Input schema: {json.dumps(t.input_schema, ensure_ascii=False, indent=2)}\n"
            f"Output schema: {json.dumps(t.output_schema or {}, ensure_ascii=False, indent=2)}"
            for t in cluster
        )

        prompt = MERGER_PROMPT.format(tools=tool_specs)

        merger_system = MERGER_SYSTEM
        if self._prompt_store:
            try:
                active = self._prompt_store.get_active("absorber_merger_prompt")
                if active:
                    merger_system = active.content
            except Exception:
                logger.warning("Failed to read active merger prompt", exc_info=True)

        try:
            source_code = await self._remote_generate(merger_system, prompt)
            source_code = self._clean_source(source_code)

            # Extract tool name from the generated code
            merged_name = self._extract_merged_name(source_code, cluster)

            return {
                "success": True,
                "merged_name": merged_name,
                "source_code": source_code,
                "description": f"Merged tool combining: {', '.join(t.name for t in cluster)}",
                "input_schema": self._merge_input_schema(cluster),
                "output_schema": self._merge_output_schema(cluster),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _heuristic_merge(self, cluster: list[ToolSpec]) -> dict[str, Any]:
        """Fallback merge: pick the tool with the broadest input schema."""
        # Sort by number of input properties (broadest first)
        sorted_tools = sorted(
            cluster,
            key=lambda t: len(t.input_schema.get("properties", {})),
            reverse=True,
        )
        best = sorted_tools[0]
        file_path = f"{self._sandbox_dir}/{best.name}.py"

        return {
            "success": True,
            "merged_name": best.name,
            "source_code": "",  # use existing file
            "description": best.description,
            "input_schema": best.input_schema,
            "output_schema": best.output_schema,
            "file_path": file_path,
        }

    @staticmethod
    def _clean_source(raw: str) -> str:
        """Strip markdown fences."""
        import re
        match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    @staticmethod
    def _extract_merged_name(source_code: str, cluster: list[ToolSpec]) -> str:
        """Try to extract tool name from the generated code, or derive one."""
        import re
        # Look for a comment like "# merged tool: xxx"
        match = re.search(r"#\s*(?:merged|tool)[\s:]+(\w+)", source_code, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Derive from cluster names: find common prefix
        names = sorted([t.name for t in cluster])
        prefix = names[0]
        for name in names[1:]:
            while not name.startswith(prefix):
                prefix = prefix[:-1]
            if not prefix:
                break

        if len(prefix) >= 4:
            return prefix.rstrip("_")

        # Fallback: join first parts
        return "_".join(n.split("_")[0] for n in names[:3])

    @staticmethod
    def _merge_input_schema(cluster: list[ToolSpec]) -> dict[str, Any]:
        """Merge input schemas: union of all properties."""
        all_props: dict[str, Any] = {}
        all_required: list[str] = []
        for t in cluster:
            props = t.input_schema.get("properties", {})
            for k, v in props.items():
                if k not in all_props:
                    all_props[k] = v
            req = t.input_schema.get("required", [])
            for r in req:
                if r not in all_required:
                    all_required.append(r)
        return {
            "type": "object",
            "properties": all_props,
            "required": all_required,
        }

    @staticmethod
    def _merge_output_schema(cluster: list[ToolSpec]) -> dict[str, Any]:
        """Merge output schemas: union of all properties."""
        all_props: dict[str, Any] = {}
        for t in cluster:
            schema = t.output_schema or {}
            props = schema.get("properties", {})
            for k, v in props.items():
                if k not in all_props:
                    all_props[k] = v
        return {
            "type": "object",
            "properties": all_props,
        }

    # -- Test runner ---------------------------------------------------------

    async def _run_parent_tests(
        self, cluster: list[ToolSpec], merged_name: str, source_code: str
    ) -> bool:
        """Run existing tool tests from parent tools against the merged tool.

        For now, this just verifies the module can be imported and InputModel
        can be instantiated.  A full implementation would store and replay
        historical tool invocations.
        """
        # Basic import + structure check
        result = self._verifier.verify(merged_name, source_code)
        return result.passed

    # -- Lineage -------------------------------------------------------------

    def _record_lineage(
        self,
        child: ToolSpec,
        parents: list[ToolSpec],
        relation: str,
    ) -> None:
        """Write lineage records to tool_lineage table."""
        conn = self._registry._conn
        if conn is None:
            return

        now = datetime.now(timezone.utc).isoformat()
        for parent in parents:
            conn.execute(
                """
                INSERT INTO tool_lineage
                    (child_tool_name, child_tool_version,
                     parent_tool_name, parent_tool_version, relation, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (child.name, child.version, parent.name, parent.version, relation, now),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

AGGREGATOR_SYSTEM = """You are a tool aggregator for a local AI assistant. Your job is to decide whether a group of tools should be merged into a single canonical tool.

RULES:
1. Only merge tools that perform the SAME BASE ACTION with the SAME ALGORITHM and return SIMILAR OUTPUT STRUCTURES.
2. Do NOT merge tools just because they have similar names.
3. Do NOT merge tools if they handle fundamentally different data types or domains.
4. Do NOT merge tools if merging would require complex detect-then-dispatch logic.
5. If tools differ only in input format or minor variations, they ARE mergeable.
6. If tools serve different user intents or have different risk levels, they are NOT mergeable.

Respond with a JSON object:
{"merge": true/false, "reason": "brief explanation"}"""

AGGREGATOR_PROMPT = """Here are the tools in this cluster:

{tools}

Should these tools be merged into a single canonical tool? Analyze their:
- Input/output schemas (are they compatible?)
- Descriptions (do they do the same thing?)
- Risk levels (are they similar?)

Respond with {{"merge": true/false, "reason": "..."}}"""

MERGER_SYSTEM = """You are a tool merger for a local AI assistant. Given multiple tool specifications, generate a single canonical Python tool that handles all of them.

RULES:
1. Output a single Python module with InputModel, OutputModel, and run().
2. The merged InputModel should accept ALL parameters from all parent tools.
3. The merged run() should detect which parent behavior is needed based on which parameters are provided.
4. Use only allowed imports: pydantic, json, re, datetime, math, pathlib, typing, collections, itertools, functools, hashlib, base64, urllib.parse, csv, io, textwrap, unicodedata.
5. NO os, sys, subprocess, shutil, socket, http, requests, httpx, ctypes, signal, eval, exec, __import__.
6. NO file I/O outside /tmp/tool_sandbox.
7. NO network access or shell commands.
8. Keep under 120 lines.
9. Include a comment at the top: "# merged tool: <name>"
10. Output ONLY the Python source code, no markdown fences."""

MERGER_PROMPT = """Here are the tools to merge:

{tools}

Generate a single canonical Python tool that replaces all of the above tools.
The merged tool should:
- Accept all input parameters from all parent tools
- Route to the correct behavior based on which parameters are provided
- Return a unified OutputModel with all possible output fields

Output ONLY the Python source code."""
