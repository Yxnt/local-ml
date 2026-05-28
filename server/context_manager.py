import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Entity:
    name: str
    type: str  # person, thing, topic
    aliases: list[str] = field(default_factory=list)
    last_mentioned: datetime = field(default_factory=datetime.now)
    context: str = ""


@dataclass
class ConversationContext:
    entities: dict[str, Entity] = field(default_factory=dict)
    recent_history: list[dict] = field(default_factory=list)
    current_topic: str = ""


# Common Chinese surnames (single character).
_SURNAMES = (
    "张|王|李|赵|刘|陈|杨|黄|周|吴|徐|孙|马|朱|胡|郭|林|何|高|罗|"
    "郑|梁|谢|宋|唐|许|韩|冯|邓|曹|彭|曾|萧|田|董|袁|潘|于|蒋|蔡|"
    "余|杜|叶|程|苏|魏|吕|丁|任|沈|姚|卢|姜|崔|钟|谭|陆|汪|范|金|"
    "石|廖|贾|夏|韦|付|方|白|邹|孟|熊|秦|邱|江|尹|薛|闫|段|雷|侯|"
    "龙|史|陶|贺|顾|毛|郝|龚|邵|万|钱|严|覃|武|戴|莫|孔|向|汤"
)

# Titles that follow a surname to form a person reference (e.g. 张总, 王经理).
_TITLES = "总|经理|老师|教授|医生|主任|院长|局长|部长|科长|博士|先生|女士|小姐|同学|师傅|前辈"

# Pattern 1: Surname + title.  This is the most reliable pattern because the
# title acts as a natural word boundary.  Example matches: 张总, 王经理, 陈老师.
_TITLE_PERSON_RE = re.compile(rf"(?:{_SURNAMES})(?:{_TITLES})")

# Pattern 2: Surname + 1-2 CJK characters that are NOT followed by another CJK
# character.  This catches actual given names (e.g. 张三, 李明) at punctuation
# or whitespace boundaries without over-matching into the next word.
# Example: "张三来了" does NOT match (来 is CJK), but "张三。" does.
_NAME_PERSON_RE = re.compile(rf"(?:{_SURNAMES})[一-鿿]{{1,2}}(?![一-鿿])")

# Quoted strings used as topic / thing references.
_QUOTED_RE = re.compile("[\u0022\u201c\u201d\u300c]([^\u0022\u201c\u201d\u300d]{1,30})[\u0022\u201d\u300d]")

# Pronoun patterns for resolution.
_PRONOUN_MAP: dict[str, str] = {
    "他": "person",
    "她": "person",
    "它": "thing",
    "这个": "thing",
    "那个": "thing",
    "他们": "person",
    "她们": "person",
}

# Common Chinese sentence particles / verbs that mark word boundaries.
# Used to trim the tail of name-pattern-2 matches when they bleed into verbs.
_BOUNDARY_CHARS = set("的了着过呢吧吗啊是有没有说来看去做给把被从在和与或但")


def extract_entities(text: str) -> dict[str, Entity]:
    """Extract entities from text using simple heuristics.

    Returns a dict keyed by canonical entity name.
    """
    entities: dict[str, Entity] = {}
    now = datetime.now()

    # 1. Title-based person references (highest confidence).
    for match in _TITLE_PERSON_RE.finditer(text):
        name = match.group(0)
        if len(name) < 2:
            continue
        entities[name] = Entity(
            name=name,
            type="person",
            last_mentioned=now,
            context=_surrounding(text, match.start(), match.end()),
        )

    # 2. Given-name person references at word boundaries.
    for match in _NAME_PERSON_RE.finditer(text):
        name = match.group(0)
        # If already captured by title pattern, skip.
        if name in entities:
            continue
        # Trim trailing boundary characters that bled into the match.
        name = _trim_boundary(name)
        if len(name) < 2:
            continue
        if name not in entities:
            entities[name] = Entity(
                name=name,
                type="person",
                last_mentioned=now,
                context=_surrounding(text, match.start(), match.end()),
            )

    # 3. Quoted strings — treated as things or topics.
    for match in _QUOTED_RE.finditer(text):
        content = match.group(1).strip()
        if content and content not in entities:
            entities[content] = Entity(
                name=content,
                type="thing",
                last_mentioned=now,
                context=_surrounding(text, match.start(), match.end()),
            )

    return entities


def _trim_boundary(name: str) -> str:
    """Remove trailing boundary characters from a name.

    e.g. "张说了" → "张" (trims the verb "说" and particle "了").
    Only trims the last character to avoid over-trimming two-char given names.
    """
    if len(name) > 2 and name[-1] in _BOUNDARY_CHARS:
        return name[:-1]
    return name


def _surrounding(text: str, start: int, end: int, window: int = 40) -> str:
    """Return a snippet of text around the matched span."""
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return text[lo:hi].strip()


class ContextManager:
    """Manages conversational context: entities, pronouns, time, and history."""

    def __init__(self, max_history_turns: int = 10):
        self.context = ConversationContext()
        self.max_history_turns = max_history_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, user_input: str, agent_response: str) -> None:
        """Update context after each conversation turn."""
        # Extract entities from both sides of the conversation.
        new_entities = extract_entities(user_input)
        new_entities.update(extract_entities(agent_response))

        now = datetime.now()
        for key, entity in new_entities.items():
            if key in self.context.entities:
                # Refresh timestamp and context.
                existing = self.context.entities[key]
                existing.last_mentioned = now
                existing.context = entity.context
                # Merge aliases.
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
            else:
                self.context.entities[key] = entity

        # Determine current topic from the user input.
        topic = self._detect_topic(user_input)
        if topic:
            self.context.current_topic = topic

        # Maintain sliding window of recent history.
        self.context.recent_history.append({"role": "user", "content": user_input})
        self.context.recent_history.append({"role": "agent", "content": agent_response})
        if len(self.context.recent_history) > self.max_history_turns * 2:
            self.context.recent_history = self.context.recent_history[-self.max_history_turns * 2 :]

    def resolve_references(self, text: str) -> str:
        """Replace pronouns with the most recently mentioned entity of the right type."""
        if not self.context.entities:
            return text

        result = text
        # Sort pronouns by length descending so longer ones (他们) match before
        # shorter ones (他) to avoid partial replacement.
        for pronoun, expected_type in sorted(_PRONOUN_MAP.items(), key=lambda kv: -len(kv[0])):
            if pronoun not in result:
                continue
            candidate = self._most_recent_entity(expected_type)
            if candidate:
                result = result.replace(pronoun, candidate.name, 1)
        return result

    def resolve_time(self, text: str) -> tuple[datetime, datetime]:
        """Extract a (start, end) datetime range from time expressions in *text*.

        Falls back to the current day if no expression is found.
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        patterns: list[tuple[str, tuple[datetime, datetime]]] = [
            ("今天", (today_start, today_start + timedelta(days=1))),
            ("昨天", (today_start - timedelta(days=1), today_start)),
            ("前天", (today_start - timedelta(days=2), today_start - timedelta(days=1))),
            ("本月", (today_start.replace(day=1), self._next_month_start(now))),
            (
                "上个月",
                (
                    self._prev_month_start(now),
                    today_start.replace(day=1),
                ),
            ),
            ("上周", self._last_week_range(now)),
        ]

        # Check longer expressions first to avoid partial matches.
        for keyword, (start, end) in patterns:
            if keyword in text:
                return (start, end)

        # Default: today.
        return (today_start, today_start + timedelta(days=1))

    def get_relevant_context(self, query: str) -> str:
        """Return a compact string of context relevant to *query*.

        Includes:
        - Current topic (if any)
        - Entities mentioned in or related to the query
        - The most recent history turns
        """
        parts: list[str] = []

        if self.context.current_topic:
            parts.append(f"当前话题: {self.context.current_topic}")

        # Resolve any pronouns in the query to identify relevant entities.
        resolved = self.resolve_references(query)
        relevant_names = [
            name
            for name in self.context.entities
            if name in resolved
        ]
        if relevant_names:
            entity_lines = []
            for name in relevant_names:
                e = self.context.entities[name]
                line = f"- {e.name} ({e.type})"
                if e.aliases:
                    line += f" 别名: {', '.join(e.aliases)}"
                entity_lines.append(line)
            parts.append("相关实体:\n" + "\n".join(entity_lines))

        # Include last few turns for continuity.
        if self.context.recent_history:
            recent = self.context.recent_history[-4:]  # last 2 turns
            history_lines = [f"{m['role']}: {m['content']}" for m in recent]
            parts.append("最近对话:\n" + "\n".join(history_lines))

        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _most_recent_entity(self, entity_type: str) -> Entity | None:
        """Return the most recently mentioned entity of the given type."""
        candidates = [
            e for e in self.context.entities.values() if e.type == entity_type
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.last_mentioned)

    def _detect_topic(self, text: str) -> str:
        """Heuristic: pick the first quoted string or first entity as topic."""
        quoted = _QUOTED_RE.search(text)
        if quoted:
            return quoted.group(1)
        entities = extract_entities(text)
        if entities:
            return next(iter(entities))
        return ""

    @staticmethod
    def _next_month_start(dt: datetime) -> datetime:
        if dt.month == 12:
            return dt.replace(year=dt.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return dt.replace(month=dt.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _prev_month_start(dt: datetime) -> datetime:
        if dt.month == 1:
            return dt.replace(year=dt.year - 1, month=12, day=1, hour=0, minute=0, second=0, microsecond=0)
        return dt.replace(month=dt.month - 1, day=1, hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _last_week_range(dt: datetime) -> tuple[datetime, datetime]:
        """Return (Monday 00:00, Sunday 00:00) of the previous week."""
        # weekday(): Monday=0 ... Sunday=6
        days_since_monday = dt.weekday()
        this_monday = dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
        last_monday = this_monday - timedelta(days=7)
        last_sunday = this_monday - timedelta(days=1)
        return (last_monday, last_sunday.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
