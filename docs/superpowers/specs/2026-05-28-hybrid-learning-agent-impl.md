# Hybrid Learning Agent - Implementation Details

## 概述

HybridAgent 是一个混合学习 Agent，结合本地模型实时处理和远程模型异步学习。它继承现有 Agent 的工具调用能力，并增加学习机制。

## 文件结构

```
server/
├── hybrid_agent.py          # 主入口，协调各组件
├── local_processor.py       # 本地模型处理
├── remote_analyzer.py       # 远程 LLM 分析
├── rule_manager.py          # 规则管理
├── learning_controller.py   # 学习控制
├── confidence_evaluator.py  # 置信度评估
└── session_sanitizer.py     # 会话级脱敏器

memory/
└── memory.py                # 扩展支持 RULE 类型

tests/
├── test_hybrid_agent.py
├── test_local_processor.py
├── test_remote_analyzer.py
├── test_rule_manager.py
├── test_learning_controller.py
└── test_session_sanitizer.py
```

## 与现有系统的关系

### 继承现有 Agent 的能力

HybridAgent 继承现有 Agent 的以下能力：
- 工具分发（Obsidian、Calendar、Email、SmartHome、Computer Use）
- 记忆系统集成（soul、user、memory）
- WebSocket 支持

### 新增能力

HybridAgent 新增以下能力：
- 双向学习（远程反馈 + 本地模式发现）
- 规则管理（动态规则类型、优先级、生命周期）
- 会话级脱敏（保证隐私）
- 置信度评估（决定是否需要远程帮助）

### 工具调用流程

```
用户输入
    ↓
本地处理 (LocalProcessor)
    ↓
模型返回工具调用?
    ├─ 是 → 执行工具 (调用现有集成)
    │        ↓
    │        用工具结果继续处理
    │        ↓
    │        评估置信度
    └─ 否 → 直接评估置信度
    ↓
置信度 < 0.9?
    ├─ 是 → 远程学习
    └─ 否 → 本地模式发现
    ↓
返回结果
```

### 集成点

| 集成 | 调用方式 | 说明 |
|------|---------|------|
| Obsidian | `obsidian_search()`, `obsidian_read()` | 笔记搜索和读取 |
| Calendar | `calendar_search()`, `calendar_upcoming()` | 日程查询 |
| Email | `email_search()`, `email_recent()` | 邮件搜索 |
| SmartHome | `smart_home_list_devices()`, `smart_home_control()` | 智能家居控制 |
| Computer Use | `computer_action()` | 屏幕控制 |
| Memory | `memory_remember()`, `memory_recall()` | 记忆管理 |

## 数据结构定义

### 1. LocalResult (本地处理结果)

```python
@dataclass
class LocalResult:
    answer: str                          # 最终回答
    confidence: float                    # 置信度 0.0-1.0
    reasoning: str                       # 推理过程
    entities_found: list[Entity]         # 找到的实体
    pronouns_resolved: dict[str, str]    # 解析的代词 {占位符: 实体名}
    time_parsed: TimeRange | None        # 解析的时间范围
    rules_applied: list[str]             # 应用的规则 ID
    ambiguity_detected: bool             # 是否检测到歧义
    needs_remote_help: bool              # 是否需要远程帮助

@dataclass
class AgentResponse:
    answer: str                          # 最终回答
    confidence: float                    # 置信度
    is_learning: bool                    # 是否正在后台学习
    learning_task: asyncio.Task | None   # 学习任务 (可用于等待)

@dataclass
class Entity:
    name: str
    type: str                            # person/thing/place/topic
    confidence: float
    source: str                          # memory/history/rule

@dataclass
class TimeRange:
    start: datetime
    end: datetime
    expression: str                      # 原始表达式 "上周"
```

### 2. RemoteFeedback (远程反馈)

```python
@dataclass
class RemoteFeedback:
    is_correct: bool
    confidence: float
    reasoning: str
    better_answer: str | None
    logic_rule: str | None
    rule_type: str | None
    corrections: list[Correction]

@dataclass
class Correction:
    field: str                           # pronoun/time/entity
    original: str
    corrected: str
    reason: str
```

### 3. LearningRule (学习规则)

```python
@dataclass
class LearningRule:
    id: str                              # rule_xxxxxxxx
    rule_type: str                       # 任何类型 (不限制)
    pattern: str                         # 触发模式 (人类可读)
    logic: str                           # 规则逻辑 (给 LLM 的指令)
    confidence: float
    source: str                          # remote_feedback/local_pattern/user_taught
    created_at: datetime
    updated_at: datetime
    usage_count: int
    success_count: int
    examples: list[RuleExample]
    status: str                          # active/inactive/archived
    metadata: dict

@dataclass
class RuleExample:
    input: str
    expected_output: str
    actual_output: str | None
    was_correct: bool
```

### 4. SanitizedContext (脱敏上下文)

```python
@dataclass
class SanitizedContext:
    user_input: str                      # 脱敏后的输入
    history: list[str]                   # 脱敏后的历史
    entities: list[dict]                 # 实体类型 (不含名称)
    local_answer: str
    confidence: float
    reasoning: str
    rules_applied: list[str]
    timestamp: datetime
```

### 5. SessionSanitizer (会话级脱敏器)

```python
@dataclass
class EntityMapping:
    original: str
    placeholder: str
    entity_type: str                     # person/thing/place/topic
    created_at: datetime

@dataclass
class SanitizedText:
    original: str
    sanitized: str
    mapping: dict[str, str]              # 占位符 → 原文
    level: SensitivityLevel

class SessionSanitizer:
    """会话级脱敏器 - 保证同一会话内实体映射一致"""

    def __init__(self):
        self.entity_mappings: dict[str, EntityMapping] = {}
        self.reverse_mappings: dict[str, EntityMapping] = {}
        self.counters: dict[str, int] = {"PERSON": 0, "THING": 0, "PLACE": 0, "TOPIC": 0}

    def sanitize(self, text: str) -> SanitizedText:
        """脱敏文本 - 复用已有映射，保持一致性"""
        # 1. 替换已知实体
        # 2. 检测并映射新实体
        # 3. 脱敏结构化数据 (邮箱、电话等)
        ...

    def desanitize(self, text: str) -> str:
        """还原占位符为原始实体"""
        ...

    def get_sanitized_history(self, history: list[str]) -> list[str]:
        """使用相同映射脱敏历史记录"""
        ...

    def get_entity_types_only(self) -> list[dict]:
        """获取实体类型 (不含名称) - 用于发送给远程"""
        ...

    # 辅助方法 (实现时补充)
    # - _detect_new_entities(text) -> list[str]
    # - _classify_entity(entity) -> str
    # - _sanitize_structured(text) -> tuple[str, dict]
    # - _detect_level(text) -> SensitivityLevel
```

## 详细实现

### 1. HybridAgent (主入口)

```python
class HybridAgent:
    """混合学习 Agent 主入口"""

    def __init__(self, config: dict):
        self.memory = MemoryManager(config["memory"])
        self.session_sanitizer = SessionSanitizer()
        self.local_model = self._init_local_model(config["local"])
        self.remote_model = self._init_remote_model(config["remote"])

        self.rule_manager = RuleManager(self.memory)
        self.local_processor = LocalProcessor(self.local_model, self.memory, self.rule_manager)
        self.remote_analyzer = RemoteAnalyzer(self.remote_model)
        self.learning_controller = LearningController(
            self.session_sanitizer, self.remote_analyzer, self.rule_manager, config["learning"]
        )

    async def run(self, user_input: str) -> AgentResponse:
        """主入口: 处理用户输入"""

        # 1. 本地处理
        result = await self.local_processor.process(user_input)

        # 2. 判断是否需要远程学习
        learning_task = None
        if result.confidence < 0.9:
            learning_task = asyncio.create_task(
                self.learning_controller.learn(user_input, result)
            )

            # 如果是连续对话，等待学习完成
            if self._is_followup_question(user_input):
                await learning_task
                learning_task = None
                result = await self.local_processor.process(user_input)

        # 3. 更新规则统计
        for rule_id in result.rules_applied:
            await self.rule_manager.update_rule_stats(rule_id, result)

        # 4. 保存对话到记忆
        await self._save_conversation(user_input, result.answer)

        return AgentResponse(
            answer=result.answer,
            confidence=result.confidence,
            is_learning=learning_task is not None,
            learning_task=learning_task
        )

    async def run_with_progress(self, user_input: str, callback: Callable[[str], None]) -> str:
        """带进度回调的入口"""
        # 类似 run()，但在关键步骤调用 callback()
        ...

    def _is_followup_question(self, user_input: str) -> bool:
        """判断是否是连续对话 (包含代词引用)"""
        pronouns = ["他", "她", "它", "这个", "那个", "这", "那"]
        return any(p in user_input for p in pronouns)

    async def _save_conversation(self, user_input: str, answer: str):
        """保存对话到记忆"""
        await self.memory.remember(
            content=f"用户: {user_input}\n助手: {answer}",
            memory_type=MemoryType.CONVERSATION,
            metadata={"timestamp": datetime.now().isoformat()}
        )

    # 辅助方法 (实现时补充)
    # - _init_local_model(config) -> ModelBackend
    # - _init_remote_model(config) -> RemoteBackend
```

### 2. LocalProcessor (本地处理器)

```python
class LocalProcessor:
    """本地模型处理器"""

    def __init__(self, model, memory, rule_manager):
        self.model = model
        self.memory = memory
        self.rule_manager = rule_manager
        self.confidence_evaluator = ConfidenceEvaluator()

    async def process(self, user_input: str) -> LocalResult:
        """处理用户输入"""

        # 1. 查询相关规则
        rules = await self.rule_manager.get_relevant_rules(user_input)

        # 2. 获取相关记忆
        memories = await self.memory.recall(user_input, limit=5)

        # 3. 获取最近对话历史
        history = await self.memory.get_recent_conversations(limit=5)

        # 4. 构建 prompt (包含规则、记忆、历史)
        prompt = self._build_prompt(user_input, rules, memories, history)

        # 5. 本地模型生成
        response = await self.model.generate_async(prompt)

        # 6. 解析响应 (JSON)
        parsed = self._parse_response(response)

        # 7. 评估置信度
        confidence = self.confidence_evaluator.evaluate(
            user_input=user_input, response=parsed, rules=rules, memories=memories
        )

        result = LocalResult(
            answer=parsed["answer"],
            confidence=confidence,
            reasoning=parsed["reasoning"],
            entities_found=parsed["entities"],
            pronouns_resolved=parsed["pronouns"],
            time_parsed=parsed["time"],
            rules_applied=[r.id for r in rules],
            ambiguity_detected=parsed["has_ambiguity"],
            needs_remote_help=confidence < 0.9
        )

        # 8. 如果成功，提取本地模式
        if confidence > 0.9:
            await self._extract_and_store_pattern(user_input, result)

        return result

    def _build_prompt(self, user_input, rules, memories, history) -> str:
        """构建本地模型的 prompt"""
        # 包含: 当前时间、用户输入、相关规则、相关记忆、最近对话
        # 要求: JSON 格式输出
        ...

    def _parse_response(self, response: str) -> dict:
        """解析本地模型的 JSON 响应"""
        # 成功: 返回解析后的 dict
        # 失败: 返回默认结构 (has_ambiguity=True)
        ...

    async def _extract_and_store_pattern(self, user_input: str, result: LocalResult):
        """从成功案例中提取模式并存储"""
        ...

    def _extract_success_pattern(self, user_input: str, result: LocalResult) -> dict | None:
        """从成功案例中提取模式"""
        # 检查代词解析、时间解析、实体匹配
        # 返回 {"type": "...", "logic": "..."} 或 None
        ...

    # 辅助方法 (实现时补充)
    # - _format_rules(rules) -> str
    # - _format_memories(memories) -> str
    # - _format_history(history) -> str
```

### 3. RemoteAnalyzer (远程分析器)

```python
class RemoteAnalyzer:
    """远程 LLM 分析器"""

    def __init__(self, remote_backend):
        self.remote = remote_backend

    async def analyze(self, context: SanitizedContext) -> RemoteFeedback:
        """分析本地处理结果"""
        prompt = self._build_analysis_prompt(context)
        response = await self.remote.generate_async(prompt)
        return self._parse_feedback(response)

    def _build_analysis_prompt(self, context: SanitizedContext) -> str:
        """构建分析 prompt"""
        # 包含: 脱敏后的输入/历史、实体类型、本地回答、置信度、推理过程
        # 要求: 判断是否正确、给出理由、提取通用规则、发现新模式
        # rule_type 可以是任何类型，不限于预定义
        ...

    def _parse_feedback(self, response: str) -> RemoteFeedback:
        """解析远程反馈 JSON"""
        # 成功: 返回 RemoteFeedback
        # 失败: 返回 RemoteFeedback(is_correct=False, confidence=0.0)
        ...
```

### 4. RuleManager (规则管理器)

```python
class RuleManager:
    """管理学习到的规则"""

    def __init__(self, memory: MemoryManager):
        self.memory = memory

    async def get_relevant_rules(self, context: str) -> list[LearningRule]:
        """获取相关规则 - 处理冲突"""
        # 1. 提取关键词
        # 2. 搜索规则 (只搜索 active 状态)
        # 3. 按优先级排序
        # 4. 去重: 同类型只保留最高优先级的
        # 5. 最多返回10条
        ...

    async def add_rule(
        self,
        rule_type: str,       # 不限制类型
        logic: str,
        confidence: float,
        source: str = "remote_feedback",
        example: dict | None = None
    ) -> LearningRule:
        """添加新规则 - 接受任何规则类型"""
        ...

    async def update_rule_stats(self, rule_id: str, result: LocalResult):
        """更新规则统计 - 基于置信度判断成功"""
        # 置信度 > 0.8 视为成功
        # 成功率 < 0.3 且使用 >= 5 次，标记为 inactive
        ...

    async def cleanup_expired_rules(self):
        """清理过期规则 (90天未使用 -> archived)"""
        ...

    async def cleanup_low_success_rules(self):
        """清理低成功率规则 (成功率 < 30% -> inactive)"""
        ...

    def _get_priority(self, rule: LearningRule) -> float:
        """计算规则优先级 - 成功率 + 使用次数加权"""
        # base_priority: user_taught=100, remote_feedback=80, local_pattern=60
        # 未测试: base * 0.6
        # 已测试: base * (success_rate * trust + 0.5 * (1 - trust))
        # trust = min(usage_count / 10, 1.0)
        ...

    # 辅助方法 (实现时补充)
    # - _extract_keywords(text) -> list[str]
    # - _extract_pattern(logic) -> str
    # - _to_learning_rule(data) -> LearningRule
```

### 5. LearningController (学习控制器)

```python
class LearningController:
    """控制何时向远程学习"""

    def __init__(self, session_sanitizer, remote_analyzer, rule_manager, config):
        self.session_sanitizer = session_sanitizer
        self.remote = remote_analyzer
        self.rule_manager = rule_manager
        self.config = config

        # 学习预算
        self.hourly_count = 0
        self.daily_count = 0
        self.last_hourly_reset = datetime.now()
        self.last_daily_reset = datetime.now()

        # 并发控制
        self._learning_lock = asyncio.Lock()
        self._pending_learnings: dict[str, asyncio.Task] = {}

    async def learn(self, user_input: str, result: LocalResult):
        """执行学习流程 - 带并发控制"""

        # 1. 生成学习任务唯一标识
        learning_key = self._generate_learning_key(user_input)

        # 2. 检查是否已有相同学习任务
        if learning_key in self._pending_learnings:
            return

        # 3. 检查预算
        if not self._check_budget():
            return

        # 4. 标记学习任务开始
        self._pending_learnings[learning_key] = asyncio.current_task()

        try:
            async with self._learning_lock:
                # 5. 准备脱敏上下文
                context = self._prepare_context(user_input, result)

                # 6. 远程分析
                feedback = await self.remote.analyze(context)

                # 7. 存储规则
                if feedback.logic_rule and feedback.confidence > 0.7:
                    await self.rule_manager.add_rule(
                        rule_type=feedback.rule_type or "general",
                        logic=feedback.logic_rule,
                        confidence=feedback.confidence,
                        example={
                            "input": context.user_input,
                            "expected": feedback.better_answer,
                            "actual": result.answer,
                            "was_correct": feedback.is_correct
                        }
                    )

                # 8. 更新预算
                self.hourly_count += 1
                self.daily_count += 1

        except Exception as e:
            logger.error(f"Learning failed: {e}")

        finally:
            self._pending_learnings.pop(learning_key, None)

    def _prepare_context(self, user_input: str, result: LocalResult) -> SanitizedContext:
        """准备脱敏上下文 - 使用会话级脱敏器保证一致性"""
        sanitized = self.session_sanitizer.sanitize(user_input)
        history = self.session_sanitizer.get_sanitized_history(self._get_recent_history())
        entity_types = self.session_sanitizer.get_entity_types_only()

        return SanitizedContext(
            user_input=sanitized.sanitized,
            history=history,
            entities=entity_types,
            local_answer=result.answer,
            confidence=result.confidence,
            reasoning=result.reasoning,
            rules_applied=result.rules_applied,
            timestamp=datetime.now()
        )

    def _check_budget(self) -> bool:
        """检查学习预算"""
        # 每小时限制、每日限制
        ...

    def _generate_learning_key(self, user_input: str) -> str:
        """生成学习任务唯一标识"""
        return hashlib.md5(user_input[:50].encode()).hexdigest()

    # 辅助方法 (实现时补充)
    # - _get_recent_history() -> list[str]
```

### 6. ConfidenceEvaluator (置信度评估器)

```python
class ConfidenceEvaluator:
    """评估本地处理的置信度"""

    def evaluate(self, user_input, response, rules, memories) -> float:
        """评估置信度 (加权平均)"""

        scores = []

        # 1. 规则匹配度 (权重 0.3)
        # 有规则: 取最高置信度; 无规则: 0.5

        # 2. 代词解析度 (权重 0.2)
        # 有代词且解析成功: 0.8; 有代词未解析: 0.4; 无代词: 0.9

        # 3. 时间解析度 (权重 0.15)
        # 有时间表达: 0.8; 无时间表达: 0.9

        # 4. 记忆匹配度 (权重 0.2)
        # 有记忆: 0.7; 无记忆: 0.5

        # 5. 歧义检测 (权重 0.15)
        # 有歧义: 0.3; 无歧义: 0.9

        # 加权平均，限制在 0.0-1.0
        ...
```

## 学习机制

### 双向学习

| 来源 | 触发条件 | 置信度 | 说明 |
|------|---------|--------|------|
| `remote_feedback` | 本地失败 (置信度 < 0.9) | 0.8-1.0 | 远程分析，质量高 |
| `local_pattern` | 本地成功 (置信度 > 0.9) | 0.6-0.7 | 本地发现，质量中 |
| `user_taught` | 用户教导 | 1.0 | 用户明确，质量最高 |

### 规则优先级

```python
base_priority = {"user_taught": 100, "remote_feedback": 80, "local_pattern": 60}

# 未测试: base * 0.6
# 已测试: base * (success_rate * trust + 0.5 * (1 - trust))
# trust = min(usage_count / 10, 1.0)
```

### 规则冲突处理

同类型规则去重，只保留最高优先级的。

### 规则生命周期

- 90天未使用 → archived
- 成功率 < 30% (使用 >= 5 次) → inactive
- 可重新激活

### 并发学习控制

- 使用 `asyncio.Lock` 顺序执行
- 使用 `_pending_learnings` 避免重复任务

## 存储扩展

### 规则在记忆中的存储格式

```python
class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"
    CONVERSATION = "conversation"
    RULE = "rule"  # 新增
```

### 规则 metadata 格式

```python
{
    "rule_id": "rule_xxxxxxxx",
    "rule_type": "任何类型",  # 不限制
    "pattern": "触发模式",
    "logic": "规则逻辑",
    "confidence": 0.8,
    "source": "remote_feedback/local_pattern/user_taught",
    "usage_count": 0,
    "success_count": 0,
    "status": "active/inactive/archived",
    "examples": [...]
}
```

### 规则格式标准化

```python
RULE_TEMPLATE = """
## 规则: {rule_type}
触发条件: {trigger_condition}
处理步骤:
1. {step_1}
2. {step_2}
3. {step_3}
示例:
- 输入: {example_input}
- 输出: {example_output}
"""
```

## 错误处理

### 本地模型错误

- ModelLoadError: 使用备用模型
- GenerationError: 返回错误信息 (confidence=0.0)
- 其他异常: 记录日志，返回通用错误

### 远程分析错误

- ConnectionError: 跳过学习 (假设本地正确)
- APIError: 跳过学习
- JSONDecodeError: 跳过学习

## 配置格式

```yaml
hybrid_agent:
  local:
    default_model: minicpm-v-4.6
    max_tokens: 2048
    temperature: 0.7

  remote:
    enabled: true
    model: mimo-v2.5-pro
    api_key_env: MIMO_API_KEY
    base_url: https://token-plan-cn.xiaomimimo.com/v1
    timeout: 30

  learning:
    enabled: true
    confidence_threshold: 0.9
    hourly_limit: 10
    daily_limit: 100
    min_remote_confidence: 0.7

  rules:
    max_rules: 1000
    expire_days: 90
    min_success_rate: 0.3

  privacy:
    sanitize_before_remote: true
    blocked_patterns: [EMAIL, PHONE, ID_CARD, BANK_CARD]
```

## 测试用例

### test_session_sanitizer.py (5 tests)
- test_consistent_entity_mapping: 同一实体映射一致
- test_multiple_entities: 多实体不同占位符
- test_desanitize: 还原占位符
- test_get_sanitized_history: 历史脱敏一致性
- test_structured_data_sanitization: 结构化数据脱敏

### test_hybrid_agent.py (4 tests)
- test_simple_query_no_learning: 简单查询不触发学习
- test_pronoun_triggers_learning: 代词查询触发学习
- test_privacy_preserved: 隐私保护
- test_run_with_progress: 进度回调

### test_rule_manager.py (5 tests)
- test_add_and_retrieve_rule: 添加和检索规则
- test_rule_priority: 规则优先级排序
- test_dynamic_rule_types: 动态规则类型
- test_rule_conflict_resolution: 规则冲突解决
- test_rule_deactivation: 规则失效

### test_local_pattern_discovery.py (3 tests)
- test_pronoun_pattern_extraction: 代词模式提取
- test_time_pattern_extraction: 时间模式提取
- test_no_pattern_for_simple_query: 简单查询不提取模式

## 实现顺序

### Phase 1: 基础框架 (Day 1)
1. `server/session_sanitizer.py` - 会话级脱敏器
2. `server/rule_manager.py` - 规则存储和查询
3. `server/confidence_evaluator.py` - 置信度评估
4. 扩展 `memory/memory.py` - 支持 RULE 类型

### Phase 2: 本地处理 (Day 2)
5. `server/local_processor.py` - 本地模型处理
6. 集成规则查询到本地处理流程
7. 测试本地处理

### Phase 3: 远程学习 (Day 3)
8. `server/remote_analyzer.py` - 远程分析
9. `server/learning_controller.py` - 学习控制
10. 集成 SessionSanitizer 到学习流程

### Phase 4: 集成测试 (Day 4)
11. `server/hybrid_agent.py` - 主入口集成
12. 端到端测试
13. 性能优化
