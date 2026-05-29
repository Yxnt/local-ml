# Hybrid Learning Agent - Implementation Details

## 文件结构

```
server/
├── hybrid_agent.py          # 主入口，协调各组件
├── local_processor.py       # 本地模型处理
├── remote_analyzer.py       # 远程 LLM 分析
├── rule_manager.py          # 规则管理
├── learning_controller.py   # 学习控制
├── confidence_evaluator.py  # 置信度评估
└── session_sanitizer.py     # 会话级脱敏器 (新增)

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

## 数据结构定义

### 1. LocalResult (本地处理结果)

```python
# server/local_processor.py

@dataclass
class LocalResult:
    """本地模型处理结果"""
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
class Entity:
    name: str                            # 实体名称
    type: str                            # person/thing/place/topic
    confidence: float                    # 匹配置信度
    source: str                          # 来源 (memory/history/rule)

@dataclass
class TimeRange:
    start: datetime
    end: datetime
    expression: str                      # 原始表达式 "上周"
```

### 2. RemoteFeedback (远程反馈)

```python
# server/remote_analyzer.py

@dataclass
class RemoteFeedback:
    """远程 LLM 反馈"""
    is_correct: bool                     # 本地回答是否正确
    confidence: float                    # 远程判断的置信度
    reasoning: str                       # 分析理由
    better_answer: str | None            # 更好的回答 (如果本地错了)
    logic_rule: str | None               # 可学习的规则
    rule_type: str | None                # 规则类型
    corrections: list[Correction]        # 具体修正

@dataclass
class Correction:
    """具体修正"""
    field: str                           # 修正的字段 (pronoun/time/entity)
    original: str                        # 本地的值
    corrected: str                       # 正确的值
    reason: str                          # 修正原因
```

### 3. LearningRule (学习规则)

```python
# server/rule_manager.py

@dataclass
class LearningRule:
    """学习到的规则"""
    id: str                              # 规则 ID (rule_xxxxxxxx)
    rule_type: str                       # 规则类型
    pattern: str                         # 触发模式 (人类可读)
    logic: str                           # 规则逻辑 (给 LLM 的指令)
    confidence: float                    # 规则置信度
    source: str                          # 来源 (remote_feedback/user_taught)
    created_at: datetime
    updated_at: datetime
    usage_count: int                     # 使用次数
    success_count: int                   # 成功次数
    examples: list[RuleExample]          # 示例
    metadata: dict                       # 额外元数据

@dataclass
class RuleExample:
    """规则示例"""
    input: str                           # 输入 (脱敏)
    expected_output: str                 # 期望输出
    actual_output: str | None            # 实际输出
    was_correct: bool                    # 是否正确

# 规则类型 - 不限制，远程 LLM 可以定义任何类型
# 常见类型示例 (仅供参考，不是枚举限制):
# - pronoun_resolution: 代词消解
# - time_parsing: 时间解析
# - entity_matching: 实体匹配
# - intent_recognition: 意图识别
# - location_query: 地点查询
# - preference_lookup: 偏好查找
# - schedule_check: 日程查询
# - comparison: 比较
# - email_compose: 邮件撰写
# - 任何远程 LLM 认为合适的类型
```

### 4. SanitizedContext (脱敏上下文)

```python
# server/hybrid_agent.py

@dataclass
class SanitizedContext:
    """脱敏后的上下文 (发送给远程)"""
    user_input: str                      # 脱敏后的输入
    history: list[str]                   # 脱敏后的历史
    entities: list[dict]                 # 实体类型 (不含名称)
    local_answer: str                    # 本地回答
    confidence: float                    # 本地置信度
    reasoning: str                       # 本地推理
    rules_applied: list[str]             # 应用的规则
    timestamp: datetime
```

### 5. SessionSanitizer (会话级脱敏器)

```python
# server/session_sanitizer.py

@dataclass
class EntityMapping:
    """实体映射"""
    original: str                        # 原始实体名
    placeholder: str                     # 占位符
    entity_type: str                     # person/thing/place
    created_at: datetime

class SessionSanitizer:
    """会话级脱敏器 - 保证同一会话内实体映射一致"""

    def __init__(self):
        self.entity_mappings: dict[str, EntityMapping] = {}  # 原始 → 映射
        self.reverse_mappings: dict[str, EntityMapping] = {}  # 占位符 → 映射
        self.counters: dict[str, int] = {                     # 计数器
            "PERSON": 0, "THING": 0, "PLACE": 0, "TOPIC": 0
        }

    def sanitize(self, text: str) -> SanitizedText:
        """脱敏文本 - 复用已有映射，保持一致性"""
        sanitized = text

        # 1. 先替换已知实体
        for original, mapping in self.entity_mappings.items():
            sanitized = sanitized.replace(original, mapping.placeholder)

        # 2. 检测并映射新实体
        new_entities = self._detect_new_entities(sanitized)
        for entity in new_entities:
            entity_type = self._classify_entity(entity)
            placeholder = f"[{entity_type}_{self.counters[entity_type]}]"
            self.counters[entity_type] += 1

            mapping = EntityMapping(
                original=entity,
                placeholder=placeholder,
                entity_type=entity_type,
                created_at=datetime.now()
            )
            self.entity_mappings[entity] = mapping
            self.reverse_mappings[placeholder] = mapping

            sanitized = sanitized.replace(entity, placeholder)

        # 3. 脱敏结构化数据 (邮箱、电话等)
        sanitized, struct_mapping = self._sanitize_structured(sanitized)

        return SanitizedText(
            original=text,
            sanitized=sanitized,
            mapping={**{m.placeholder: m.original for m in self.entity_mappings.values()},
                     **struct_mapping},
            level=self._detect_level(text)
        )

    def desanitize(self, text: str) -> str:
        """还原占位符为原始实体"""
        result = text
        for placeholder, mapping in self.reverse_mappings.items():
            result = result.replace(placeholder, mapping.original)
        return result

    def get_sanitized_history(self, history: list[str]) -> list[str]:
        """使用相同映射脱敏历史记录"""
        return [self.sanitize(msg).sanitized for msg in history]

    def get_entity_types_only(self) -> list[dict]:
        """获取实体类型 (不含名称) - 用于发送给远程"""
        return [
            {"type": m.entity_type, "placeholder": m.placeholder}
            for m in self.entity_mappings.values()
        ]
```

## 详细实现

### 1. HybridAgent (主入口)

```python
# server/hybrid_agent.py

class HybridAgent:
    """混合学习 Agent 主入口"""

    def __init__(self, config: dict):
        # 初始化各组件
        self.memory = MemoryManager(config["memory"])
        self.session_sanitizer = SessionSanitizer()  # 会话级脱敏器
        self.local_model = self._init_local_model(config["local"])
        self.remote_model = self._init_remote_model(config["remote"])

        self.rule_manager = RuleManager(self.memory)
        self.local_processor = LocalProcessor(
            self.local_model,
            self.memory,
            self.rule_manager
        )
        self.remote_analyzer = RemoteAnalyzer(self.remote_model)
        self.learning_controller = LearningController(
            self.session_sanitizer,
            self.remote_analyzer,
            self.rule_manager,
            config["learning"]
        )

    async def run(self, user_input: str) -> str:
        """主入口: 处理用户输入"""

        # 1. 本地处理
        result = await self.local_processor.process(user_input)

        # 2. 判断是否需要远程学习
        if result.confidence < 0.9:
            # 告诉用户正在分析
            yield "正在分析这个问题..."

            # 异步学习
            learning_task = asyncio.create_task(
                self.learning_controller.learn(user_input, result)
            )

            # 如果是连续对话，等待学习完成
            if self._is_followup_question(user_input):
                await learning_task
                # 用新规则重新处理
                result = await self.local_processor.process(user_input)

        # 3. 更新规则统计
        for rule_id in result.rules_applied:
            await self.rule_manager.update_rule_stats(rule_id, result)

        # 4. 保存对话到记忆
        await self._save_conversation(user_input, result.answer)

        return result.answer

    def _is_followup_question(self, user_input: str) -> bool:
        """判断是否是连续对话"""
        # 检查是否有代词引用
        pronouns = ["他", "她", "它", "这个", "那个", "这", "那"]
        return any(p in user_input for p in pronouns)

    async def _save_conversation(self, user_input: str, answer: str):
        """保存对话到记忆"""
        await self.memory.remember(
            content=f"用户: {user_input}\n助手: {answer}",
            memory_type=MemoryType.CONVERSATION,
            metadata={"timestamp": datetime.now().isoformat()}
        )
```

### 2. LocalProcessor (本地处理器)

```python
# server/local_processor.py

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

        # 4. 构建 prompt
        prompt = self._build_prompt(user_input, rules, memories, history)

        # 5. 本地模型生成
        response = await self.model.generate_async(prompt)

        # 6. 解析响应
        parsed = self._parse_response(response)

        # 7. 评估置信度
        confidence = self.confidence_evaluator.evaluate(
            user_input=user_input,
            response=parsed,
            rules=rules,
            memories=memories
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

    async def _extract_and_store_pattern(self, user_input: str, result: LocalResult):
        """从成功案例中提取模式并存储"""

        pattern = self._extract_success_pattern(user_input, result)
        if pattern:
            await self.rule_manager.add_rule(
                rule_type=pattern["type"],
                logic=pattern["logic"],
                confidence=0.7,  # 本地发现的规则置信度较低
                source="local_pattern",
                example={
                    "input": user_input,
                    "output": result.answer
                }
            )

    def _extract_success_pattern(self, user_input: str, result: LocalResult) -> dict | None:
        """从成功案例中提取模式"""

        # 检查是否有代词解析
        if result.pronouns_resolved:
            return {
                "type": "pronoun_resolution",
                "logic": "当用户输入包含代词时，查找最近提到的实体"
            }

        # 检查是否有时间解析
        if result.time_parsed:
            return {
                "type": "time_parsing",
                "logic": "当用户输入包含时间表达时，转换为具体日期范围"
            }

        # 检查是否有实体匹配
        if len(result.entities_found) > 0:
            return {
                "type": "entity_matching",
                "logic": "当用户输入包含实体时，从记忆中查找相关信息"
            }

        return None

    def _build_prompt(
        self,
        user_input: str,
        rules: list[LearningRule],
        memories: list[Memory],
        history: list[dict]
    ) -> str:
        """构建本地模型的 prompt"""

        rules_text = self._format_rules(rules)
        memories_text = self._format_memories(memories)
        history_text = self._format_history(history)

        return f"""你是一个个人 AI 助手。请基于以下信息处理用户输入。

## 当前时间
{datetime.now().isoformat()}

## 用户输入
{user_input}

## 相关规则 (从历史学习)
{rules_text}

## 相关记忆
{memories_text}

## 最近对话
{history_text}

## 处理要求
1. 识别代词 (他/她/它/这个/那个) 并根据上下文解析
2. 识别时间表达 (上周/昨天/三天前) 并转换为具体日期
3. 从记忆中查找相关信息
4. 生成简洁的回答

## 输出格式
请以 JSON 格式返回:
{{
  "answer": "你的回答",
  "reasoning": "推理过程",
  "entities": [{{"name": "实体名", "type": "person/thing", "confidence": 0.9}}],
  "pronouns": {{"[PERSON_0]": "张总"}},
  "time": {{"expression": "上周", "start": "2024-01-01", "end": "2024-01-07"}} 或 null,
  "has_ambiguity": false
}}"""

    def _parse_response(self, response: str) -> dict:
        """解析本地模型的响应"""
        try:
            # 尝试解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # 如果解析失败，返回默认结构
        return {
            "answer": response,
            "reasoning": "无法解析结构化响应",
            "entities": [],
            "pronouns": {},
            "time": None,
            "has_ambiguity": True
        }
```

### 3. RemoteAnalyzer (远程分析器)

```python
# server/remote_analyzer.py

class RemoteAnalyzer:
    """远程 LLM 分析器"""

    def __init__(self, remote_backend, sanitizer):
        self.remote = remote_backend
        self.sanitizer = sanitizer

    async def analyze(self, context: SanitizedContext) -> RemoteFeedback:
        """分析本地处理结果"""

        prompt = self._build_analysis_prompt(context)
        response = await self.remote.generate_async(prompt)

        return self._parse_feedback(response)

    def _build_analysis_prompt(self, context: SanitizedContext) -> str:
        """构建分析 prompt"""

        return f"""分析以下对话处理是否正确，并给出改进建议。

## 用户输入 (已脱敏)
{context.user_input}

## 对话历史 (已脱敏)
{chr(10).join(context.history)}

## 实体类型 (不含具体名称)
{json.dumps(context.entities, ensure_ascii=False)}

## 本地回答
{context.local_answer}

## 本地置信度
{context.confidence}

## 本地推理过程
{context.reasoning}

## 应用的规则
{chr(10).join(context.rules_applied)}

## 分析要求
1. 判断本地回答是否正确
2. 分析推理过程是否有逻辑错误
3. 如果有错误，给出正确的回答
4. 提取可以通用化的规则

## 输出格式
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "详细分析",
  "better_answer": "如果本地错了，正确答案是什么",
  "logic_rule": "可以学习的通用规则 (用自然语言描述)",
  "rule_type": "你认为这个规则属于什么类型 (可以是任何类型)",
  "corrections": [
    {{
      "field": "pronoun/time/entity",
      "original": "本地的值",
      "corrected": "正确的值",
      "reason": "修正原因"
    }}
  ]
}}

注意:
- 只基于提供的脱敏数据分析，不要假设未提供的信息
- 规则要通用化，不要针对具体实体
- 如果信息不足以判断，说明需要什么额外信息
- rule_type 可以是任何你认为合适的类型，不限于预定义的类型
  例如: "location_query", "preference_lookup", "schedule_check", "comparison", "email_compose" 等
- 鼓励发现新的模式和规则类型"""

    def _parse_feedback(self, response: str) -> RemoteFeedback:
        """解析远程反馈"""
        try:
            data = json.loads(response)
            return RemoteFeedback(
                is_correct=data.get("is_correct", False),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                better_answer=data.get("better_answer"),
                logic_rule=data.get("logic_rule"),
                rule_type=data.get("rule_type"),
                corrections=[
                    Correction(**c) for c in data.get("corrections", [])
                ]
            )
        except (json.JSONDecodeError, KeyError) as e:
            return RemoteFeedback(
                is_correct=False,
                confidence=0.0,
                reasoning=f"Failed to parse feedback: {e}",
                better_answer=None,
                logic_rule=None,
                rule_type=None,
                corrections=[]
            )
```

### 4. RuleManager (规则管理器)

```python
# server/rule_manager.py

class RuleManager:
    """管理学习到的规则"""

    def __init__(self, memory: MemoryManager):
        self.memory = memory

    async def get_relevant_rules(self, context: str) -> list[LearningRule]:
        """获取相关规则"""

        # 1. 提取关键词
        keywords = self._extract_keywords(context)

        # 2. 搜索规则
        raw_rules = await self.memory.search(
            query=" ".join(keywords),
            memory_type=MemoryType.RULE,
            limit=10
        )

        # 3. 转换为 LearningRule 对象
        rules = [self._to_learning_rule(r) for r in raw_rules]

        # 4. 按优先级排序
        return sorted(rules, key=lambda r: self._get_priority(r), reverse=True)

    async def add_rule(
        self,
        rule_type: str,  # 不限制类型，远程 LLM 可以定义任何类型
        logic: str,
        confidence: float,
        source: str = "remote_feedback",  # 来源: remote_feedback/local_pattern/user_taught
        example: dict | None = None
    ) -> LearningRule:
        """添加新规则 - 接受任何规则类型"""

        # 不检查 rule_type 是否在预定义列表中
        # 远程 LLM 或本地可以创建任何新类型

        rule_id = f"rule_{uuid.uuid4().hex[:8]}"

        rule_data = {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "pattern": self._extract_pattern(logic),
            "logic": logic,
            "confidence": confidence,
            "source": source,
            "usage_count": 0,
            "success_count": 0,
            "examples": [example] if example else [],
            "status": "active",
        }

        await self.memory.remember(
            content=logic,
            memory_type=MemoryType.RULE,
            metadata=rule_data
        )

        return self._to_learning_rule(rule_data)

    async def update_rule_stats(self, rule_id: str, result: LocalResult):
        """更新规则统计 - 基于置信度判断成功"""
        # 从记忆中获取规则
        rules = await self.memory.search(
            query=rule_id,
            memory_type=MemoryType.RULE,
            limit=1
        )

        if rules:
            rule = rules[0]
            rule.metadata["usage_count"] += 1

            # 根据置信度判断是否成功
            if result.confidence > 0.8:
                rule.metadata["success_count"] += 1

            # 更新状态
            success_rate = rule.metadata["success_count"] / rule.metadata["usage_count"]
            if success_rate < 0.3 and rule.metadata["usage_count"] >= 5:
                rule.metadata["status"] = "inactive"  # 成功率太低，标记为失效

            await self.memory.update_memory(rule.id, metadata=rule.metadata)

    def _get_priority(self, rule: LearningRule) -> float:
        """计算规则优先级 - 成功率 + 使用次数加权"""
        base_priority = {
            "user_taught": 100,
            "remote_feedback": 80,
            "local_pattern": 60,
        }.get(rule.source, 40)

        # 未测试的规则打折
        if rule.usage_count == 0:
            return base_priority * 0.6

        # 使用次数越多，越信任成功率
        trust = min(rule.usage_count / 10, 1.0)  # 用10次就完全信任
        success_rate = rule.success_count / rule.usage_count

        # 加权成功率
        effective_rate = success_rate * trust + 0.5 * (1 - trust)

        return base_priority * effective_rate

    def _extract_keywords(self, text: str) -> list[str]:
        """提取关键词用于规则匹配"""
        # 简单实现：提取中文词和英文词
        keywords = []
        # 中文词 (2-4字)
        keywords.extend(re.findall(r'[一-龥]{2,4}', text))
        # 英文词
        keywords.extend(re.findall(r'[a-zA-Z]+', text))
        return keywords[:10]  # 最多10个关键词
```

### 5. LearningController (学习控制器)

```python
# server/learning_controller.py

class LearningController:
    """控制何时向远程学习"""

    def __init__(
        self,
        session_sanitizer: SessionSanitizer,
        remote_analyzer: RemoteAnalyzer,
        rule_manager: RuleManager,
        config: dict
    ):
        self.session_sanitizer = session_sanitizer
        self.remote = remote_analyzer
        self.rule_manager = rule_manager
        self.config = config

        # 学习预算
        self.hourly_count = 0
        self.daily_count = 0
        self.last_hourly_reset = datetime.now()
        self.last_daily_reset = datetime.now()

    async def learn(self, user_input: str, result: LocalResult):
        """执行学习流程"""

        # 1. 检查预算
        if not self._check_budget():
            return

        try:
            # 2. 准备脱敏上下文 (使用会话级脱敏器)
            context = self._prepare_context(user_input, result)

            # 3. 远程分析
            feedback = await self.remote.analyze(context)

            # 4. 存储规则
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

            # 5. 更新预算
            self.hourly_count += 1
            self.daily_count += 1

        except Exception as e:
            # 学习失败不影响主流程
            print(f"Learning failed: {e}")

    def _check_budget(self) -> bool:
        """检查学习预算"""
        now = datetime.now()

        # 重置小时计数
        if now - self.last_hourly_reset > timedelta(hours=1):
            self.hourly_count = 0
            self.last_hourly_reset = now

        # 重置日计数
        if now - self.last_daily_reset > timedelta(days=1):
            self.daily_count = 0
            self.last_daily_reset = now

        # 检查限制
        if self.hourly_count >= self.config.get("hourly_limit", 10):
            return False
        if self.daily_count >= self.config.get("daily_limit", 100):
            return False

        return True

    def _prepare_context(self, user_input: str, result: LocalResult) -> SanitizedContext:
        """准备脱敏上下文 - 使用会话级脱敏器保证一致性"""

        # 使用会话级脱敏器 (保证实体映射一致)
        sanitized = self.session_sanitizer.sanitize(user_input)

        # 获取脱敏后的历史 (使用相同映射)
        history = self.session_sanitizer.get_sanitized_history(
            self._get_recent_history()
        )

        # 获取实体类型 (不含名称)
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
```

### 6. ConfidenceEvaluator (置信度评估器)

```python
# server/confidence_evaluator.py

class ConfidenceEvaluator:
    """评估本地处理的置信度"""

    def evaluate(
        self,
        user_input: str,
        response: dict,
        rules: list[LearningRule],
        memories: list[Memory]
    ) -> float:
        """评估置信度"""

        scores = []

        # 1. 规则匹配度
        if rules:
            rule_score = max(r.confidence for r in rules)
            scores.append(("rule", rule_score, 0.3))
        else:
            scores.append(("rule", 0.5, 0.3))

        # 2. 代词解析度
        pronouns = response.get("pronouns", {})
        if pronouns:
            pronoun_score = 0.8 if all(v for v in pronouns.values()) else 0.4
            scores.append(("pronoun", pronoun_score, 0.2))
        else:
            scores.append(("pronoun", 0.9, 0.2))  # 没有代词，置信度高

        # 3. 时间解析度
        time_data = response.get("time")
        if time_data:
            time_score = 0.8  # 有时间表达
            scores.append(("time", time_score, 0.15))
        else:
            scores.append(("time", 0.9, 0.15))

        # 4. 记忆匹配度
        if memories:
            memory_score = 0.7
            scores.append(("memory", memory_score, 0.2))
        else:
            scores.append(("memory", 0.5, 0.2))

        # 5. 歧义检测
        has_ambiguity = response.get("has_ambiguity", False)
        ambiguity_score = 0.3 if has_ambiguity else 0.9
        scores.append(("ambiguity", ambiguity_score, 0.15))

        # 加权平均
        total_score = sum(score * weight for _, score, weight in scores)
        return min(max(total_score, 0.0), 1.0)
```

## 学习机制

### 双向学习

系统支持两种学习来源：

| 来源 | 触发条件 | 置信度 | 说明 |
|------|---------|--------|------|
| `remote_feedback` | 本地失败 (置信度 < 0.9) | 0.8-1.0 | 远程分析，质量高 |
| `local_pattern` | 本地成功 (置信度 > 0.9) | 0.6-0.7 | 本地发现，质量中 |
| `user_taught` | 用户教导 | 1.0 | 用户明确，质量最高 |

### 本地模式发现

当本地模型成功处理复杂查询时，自动提取成功模式：

```python
async def _extract_and_store_pattern(self, user_input: str, result: LocalResult):
    """从成功案例中提取模式并存储"""

    pattern = self._extract_success_pattern(user_input, result)
    if pattern:
        await self.rule_manager.add_rule(
            rule_type=pattern["type"],
            logic=pattern["logic"],
            confidence=0.7,  # 本地发现的规则置信度较低
            source="local_pattern",
            example={
                "input": user_input,
                "output": result.answer
            }
        )

def _extract_success_pattern(self, user_input: str, result: LocalResult) -> dict | None:
    """从成功案例中提取模式"""

    # 检查是否有代词解析
    if result.pronouns_resolved:
        return {
            "type": "pronoun_resolution",
            "logic": "当用户输入包含代词时，查找最近提到的实体"
        }

    # 检查是否有时间解析
    if result.time_parsed:
        return {
            "type": "time_parsing",
            "logic": "当用户输入包含时间表达时，转换为具体日期范围"
        }

    # 检查是否有实体匹配
    if len(result.entities_found) > 0:
        return {
            "type": "entity_matching",
            "logic": "当用户输入包含实体时，从记忆中查找相关信息"
        }

    return None
```

### 规则优先级

```python
def _get_priority(self, rule: LearningRule) -> float:
    """计算规则优先级 - 成功率 + 使用次数加权"""
    base_priority = {
        "user_taught": 100,      # 用户教导，最高优先级
        "remote_feedback": 80,   # 远程反馈，高优先级
        "local_pattern": 60,     # 本地发现，中优先级
    }.get(rule.source, 40)

    # 未测试的规则打折
    if rule.usage_count == 0:
        return base_priority * 0.6

    # 使用次数越多，越信任成功率
    trust = min(rule.usage_count / 10, 1.0)  # 用10次就完全信任
    success_rate = rule.success_count / rule.usage_count

    # 加权成功率
    effective_rate = success_rate * trust + 0.5 * (1 - trust)

    return base_priority * effective_rate
```

## 存储扩展

### 规则格式标准化

规则使用结构化格式，便于小模型理解和应用：

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

# 示例: 代词消解规则
rule = """
## 规则: 代词消解
触发条件: 用户输入包含"他/她/它/这个/那个"
处理步骤:
1. 查找最近5轮对话中提到的人物/事物
2. 选择最近提到的作为指代对象
3. 如果有多个候选，选择与当前话题最相关的
示例:
- 输入: "张总说下周要开会" → "他说什么了？"
- 输出: "张总说下周要开会"
"""
```

### 规则在记忆中的存储格式

```python
# memory/memory.py 扩展

class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"
    CONVERSATION = "conversation"
    RULE = "rule"  # 新增: 规则类型

# 规则存储的 metadata 格式
RULE_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "rule_id": {"type": "string"},
        "rule_type": {"type": "string"},  # 不限制类型，远程 LLM 可以定义任何类型
        "pattern": {"type": "string"},
        "logic": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "source": {"type": "string"},
        "usage_count": {"type": "integer"},
        "success_count": {"type": "integer"},
        "status": {"type": "string", "enum": ["active", "inactive", "archived"]},
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "expected_output": {"type": "string"},
                    "actual_output": {"type": "string"},
                    "was_correct": {"type": "boolean"}
                }
            }
        }
    },
    "required": ["rule_id", "rule_type", "logic", "confidence"]
}
```

## 错误处理

### 本地模型错误

```python
async def process(self, user_input: str) -> LocalResult:
    try:
        # ... 正常处理
    except ModelLoadError:
        # 模型加载失败，使用备用模型
        return await self._process_with_fallback(user_input)
    except GenerationError:
        # 生成失败，返回错误信息
        return LocalResult(
            answer="抱歉，处理您的请求时出现错误。",
            confidence=0.0,
            reasoning="Generation failed",
            # ...
        )
    except Exception as e:
        # 未知错误
        logger.error(f"Unexpected error: {e}")
        return LocalResult(
            answer="抱歉，出现了一个意外错误。",
            confidence=0.0,
            reasoning=str(e),
            # ...
        )
```

### 远程分析错误

```python
async def analyze(self, context: SanitizedContext) -> RemoteFeedback:
    try:
        # ... 正常分析
    except ConnectionError:
        # 网络错误，跳过学习
        return RemoteFeedback(
            is_correct=True,  # 假设本地正确
            confidence=0.0,
            reasoning="Remote service unavailable",
            # ...
        )
    except APIError as e:
        # API 错误
        return RemoteFeedback(
            is_correct=True,
            confidence=0.0,
            reasoning=f"API error: {e}",
            # ...
        )
    except json.JSONDecodeError:
        # 响应解析失败
        return RemoteFeedback(
            is_correct=True,
            confidence=0.0,
            reasoning="Failed to parse remote response",
            # ...
        )
```

## 配置格式

```yaml
# config.yaml - hybrid_agent 部分

hybrid_agent:
  # 本地模型配置
  local:
    default_model: minicpm-v-4.6
    max_tokens: 2048
    temperature: 0.7

  # 远程模型配置
  remote:
    enabled: true
    model: mimo-v2.5-pro
    api_key_env: MIMO_API_KEY
    base_url: https://token-plan-cn.xiaomimimo.com/v1
    timeout: 30

  # 学习配置
  learning:
    enabled: true
    confidence_threshold: 0.9
    hourly_limit: 10
    daily_limit: 100
    min_remote_confidence: 0.7

  # 规则配置
  rules:
    max_rules: 1000
    expire_days: 90
    min_success_rate: 0.3
    cleanup_interval_hours: 24

  # 隐私配置
  privacy:
    sanitize_before_remote: true
    blocked_patterns:
      - EMAIL
      - PHONE
      - ID_CARD
      - BANK_CARD
    allowed_patterns:
      - TIME_EXPRESSION
      - INTENT
      - CONFIDENCE
```

## 测试用例

### test_hybrid_agent.py

```python
class TestHybridAgent:
    """混合 Agent 主测试"""

    async def test_simple_query_no_learning(self):
        """简单查询不需要远程学习"""
        agent = HybridAgent(test_config)
        result = await agent.run("今天天气怎么样")
        assert result  # 有回答
        # 不应该触发远程学习 (无代词、无时间)

    async def test_pronoun_triggers_learning(self):
        """代词查询触发远程学习"""
        agent = HybridAgent(test_config)
        # 先建立上下文
        await agent.run("张总说下周要开会")
        # 这个应该触发学习
        result = await agent.run("他说什么了")
        assert result

    async def test_privacy_preserved(self):
        """确保隐私保护"""
        agent = HybridAgent(test_config)
        # 包含敏感信息
        result = await agent.run("张总的邮箱是zhang@test.com")
        # 验证远程收到的是脱敏后的数据
        # ...
```

### test_rule_manager.py

```python
class TestRuleManager:
    """规则管理器测试"""

    async def test_add_and_retrieve_rule(self):
        """添加和检索规则"""
        manager = RuleManager(test_memory)
        rule = await manager.add_rule(
            rule_type="pronoun_resolution",
            logic="'那个东西'指代最近讨论的话题",
            confidence=0.8
        )
        assert rule.id.startswith("rule_")

        # 检索
        rules = await manager.get_relevant_rules("那个东西")
        assert len(rules) > 0
        assert rules[0].id == rule.id

    async def test_rule_priority(self):
        """规则优先级排序"""
        manager = RuleManager(test_memory)

        # 添加不同优先级的规则
        await manager.add_rule("pronoun_resolution", "规则1", 0.9)
        await manager.add_rule("pronoun_resolution", "规则2", 0.5)

        rules = await manager.get_relevant_rules("代词")
        assert rules[0].confidence > rules[1].confidence
```

## 实现顺序

### Phase 1: 基础框架 (Day 1)
1. `server/session_sanitizer.py` - 会话级脱敏器 (保证实体映射一致)
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
