# Hybrid Learning Agent Design

## Goal

构建一个隐私优先的个人 AI Agent，通过本地模型实时处理 + 远程模型异步学习的混合架构，在保护隐私的同时持续提升能力。

## 核心理念

```
本地模型 = 学生 (实时处理，保护隐私)
远程模型 = 老师 (异步指导，提升能力)
记忆系统 = 笔记本 (存储规则，积累经验)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户入口                                  │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│   手机 App   │    CLI       │  Discord/TG  │    Web UI          │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬────────────┘
       │              │              │               │
       └──────────────┴──────┬───────┴───────────────┘
                             │
                    WebSocket / HTTP
                             │
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Agent (核心)                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LocalProcessor (本地处理)                                │   │
│  │ - 代词消解 (基于规则 + 记忆)                             │   │
│  │ - 时间解析 (正则 + 上下文)                               │   │
│  │ - 实体匹配 (记忆系统)                                    │   │
│  │ - 回答生成 (本地模型)                                    │   │
│  │ - 置信度评估                                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────┴──────────────────────────────┐   │
│  │ LearningController (学习控制)                            │   │
│  │ - 判断是否需要远程帮助                                   │   │
│  │ - 脱敏打包上下文                                         │   │
│  │ - 频率控制                                               │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────┴──────────────────────────────┐   │
│  │ RemoteAnalyzer (远程分析)                                │   │
│  │ - 发送脱敏上下文给远程 LLM                               │   │
│  │ - 接收判断 + 理由 + 规则                                 │   │
│  │ - 返回分析结果                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ RuleManager (规则管理)                                   │   │
│  │ - 存储规则到记忆                                         │   │
│  │ - 查询相关规则                                           │   │
│  │ - 规则优先级排序                                         │   │
│  │ - 规则冲突解决                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
┌──────┴──────┐       ┌──────┴──────┐       ┌──────┴──────┐
│ 记忆系统    │       │ 本地模型    │       │ 远程模型    │
│             │       │ (MLX)       │       │ (API)       │
│ - Soul      │       │ - MiniCPM   │       │ - MiMo      │
│ - User      │       │ - Gemma     │       │ - GPT-4o    │
│ - Memory    │       │ - MiniCPM-V │       │             │
│ - Rules     │       │             │       │             │
└─────────────┘       └─────────────┘       └─────────────┘
```

## Data Flow

### 实时请求流程

```
用户输入: "张总上周说的那个东西"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 本地处理                                            │
│                                                             │
│ 1. 查询相关规则                                             │
│    rules = memory.search_rules("代词消解")                  │
│                                                             │
│ 2. 应用规则处理                                             │
│    - 识别 "那个东西" 是代词                                 │
│    - 应用规则: "那个东西" 指代最近讨论的话题                 │
│    - 查找记忆: 最近讨论的话题 = "项目报告"                  │
│    - 识别 "上周" 是时间表达                                 │
│    - 解析时间: 2024-01-01 ~ 2024-01-07                      │
│                                                             │
│ 3. 生成回答                                                 │
│    answer = "张总上周说的是项目报告"                         │
│                                                             │
│ 4. 评估置信度                                               │
│    confidence = 0.85 (有代词，有时间，但规则明确)           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 判断是否需要远程学习                                │
│                                                             │
│ if confidence < 0.9:                                        │
│     # 置信度不够，需要远程帮助                              │
│     async learn_from_remote(user_input, local_result)       │
│                                                             │
│ return answer  # 先返回本地结果给用户                       │
└─────────────────────────────────────────────────────────────┘
```

### 异步学习流程

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 脱敏打包                                            │
│                                                             │
│ 原始上下文:                                                 │
│ {                                                           │
│   "user_input": "张总上周说的那个东西",                      │
│   "history": [                                              │
│     "用户: 张总说下周要开会",                               │
│     "助手: 好的，已记录"                                    │
│   ],                                                        │
│   "entities": {"张总": "person"},                           │
│   "local_answer": "张总上周说的是项目报告",                  │
│   "confidence": 0.85                                        │
│ }                                                           │
│                                                             │
│ 脱敏后:                                                     │
│ {                                                           │
│   "user_input": "[PERSON_0]上周说的那个东西",                │
│   "history": [                                              │
│     "用户: [PERSON_0]说下周要开会",                         │
│     "助手: 好的，已记录"                                    │
│   ],                                                        │
│   "entities": {"[PERSON_0]": "person"},                     │
│   "local_answer": "[PERSON_0]上周说的是项目报告",            │
│   "confidence": 0.85                                        │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 远程 LLM 分析                                       │
│                                                             │
│ Prompt:                                                     │
│ """                                                         │
│ 分析以下对话处理是否正确。                                  │
│                                                             │
│ 用户输入 (已脱敏): {sanitized_input}                        │
│ 对话历史 (已脱敏): {sanitized_history}                      │
│ 本地回答: {local_answer}                                    │
│ 本地置信度: {confidence}                                    │
│                                                             │
│ 请分析:                                                     │
│ 1. 本地回答是否正确?                                        │
│ 2. 本地的处理逻辑有什么问题?                                │
│ 3. 给出一个更好的处理规则                                   │
│                                                             │
│ 返回 JSON:                                                  │
│ {                                                           │
│   "is_correct": true/false,                                 │
│   "confidence": 0.0-1.0,                                    │
│   "reasoning": "你的分析",                                  │
│   "better_answer": "如果本地错了，正确答案是什么",           │
│   "logic_rule": "可以学习的通用规则"                        │
│ }                                                           │
│ """                                                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 存储规则                                            │
│                                                             │
│ if feedback["logic_rule"]:                                  │
│     memory.remember(                                        │
│         content=feedback["logic_rule"],                      │
│         type=MemoryType.RULE,                               │
│         metadata={                                          │
│             "rule_type": "pronoun_resolution",              │
│             "confidence": feedback["confidence"],           │
│             "source": "remote_feedback",                    │
│             "example": {                                    │
│                 "input": sanitized_input,                   │
│                 "output": feedback["better_answer"]         │
│             }                                               │
│         }                                                   │
│     )                                                       │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Rule Format (规则格式)

```python
@dataclass
class LearningRule:
    """从远程学习到的规则"""
    id: str                          # 规则 ID
    pattern: str                     # 触发模式
    rule_type: str                   # 规则类型
    logic: str                       # 规则逻辑
    examples: list[str]              # 示例
    confidence: float                # 置信度
    source: str                      # 来源
    created_at: datetime
    usage_count: int                 # 使用次数
    success_count: int               # 成功次数

# 规则类型
RULE_TYPES = {
    "pronoun_resolution": "代词消解",
    "time_parsing": "时间解析",
    "entity_matching": "实体匹配",
    "intent_recognition": "意图识别",
    "disambiguation": "歧义处理",
}
```

### 2. LocalProcessor (本地处理器)

```python
class LocalProcessor:
    """本地模型处理器"""

    def __init__(self, model, memory, rule_manager):
        self.model = model
        self.memory = memory
        self.rule_manager = rule_manager

    async def process(self, user_input: str) -> LocalResult:
        # 1. 查询相关规则
        rules = await self.rule_manager.get_relevant_rules(user_input)

        # 2. 构建 prompt (包含规则)
        prompt = self._build_prompt(user_input, rules)

        # 3. 本地模型生成
        response = await self.model.generate(prompt)

        # 4. 评估置信度
        confidence = self._evaluate_confidence(response, rules)

        return LocalResult(
            answer=response,
            confidence=confidence,
            rules_used=rules,
            reasoning=self._extract_reasoning(response)
        )

    def _build_prompt(self, user_input: str, rules: list) -> str:
        rules_text = "\n".join(f"- {r.logic}" for r in rules)
        return f"""基于以下规则处理用户输入。

用户输入: {user_input}
当前时间: {datetime.now().isoformat()}

相关规则:
{rules_text}

请:
1. 识别代词和时间表达
2. 应用规则解析
3. 生成回答
"""
```

### 3. LearningController (学习控制器)

```python
class LearningController:
    """控制何时向远程学习"""

    def __init__(self, sanitizer, remote_analyzer, rule_manager):
        self.sanitizer = sanitizer
        self.remote = remote_analyzer
        self.rule_manager = rule_manager
        self.budget = LearningBudget()

    async def maybe_learn(self, user_input: str, result: LocalResult):
        """判断是否需要学习"""

        # 检查预算
        if not self.budget.can_learn():
            return

        # 检查是否有必要
        if result.confidence > 0.9:
            return  # 本地很确定，不需要

        if not self._should_learn(result):
            return

        # 异步学习
        asyncio.create_task(
            self._learn_from_remote(user_input, result)
        )

    async def _learn_from_remote(self, user_input: str, result: LocalResult):
        # 1. 脱敏
        sanitized = self.sanitizer.sanitize(user_input)

        # 2. 打包上下文
        context = {
            "user_input": sanitized.sanitized,
            "history": self._get_sanitized_history(),
            "entities": self._get_sanitized_entities(),
            "local_answer": result.answer,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

        # 3. 远程分析
        feedback = await self.remote.analyze(context)

        # 4. 存储规则
        if feedback.get("logic_rule"):
            await self.rule_manager.add_rule(
                rule_type=self._infer_rule_type(feedback),
                logic=feedback["logic_rule"],
                confidence=feedback["confidence"],
                example={
                    "input": sanitized.sanitized,
                    "output": feedback.get("better_answer")
                }
            )
```

### 4. RuleManager (规则管理器)

```python
class RuleManager:
    """管理学习到的规则"""

    def __init__(self, memory):
        self.memory = memory

    async def get_relevant_rules(self, context: str) -> list[LearningRule]:
        """获取相关规则"""
        # 1. 提取关键词
        keywords = self._extract_keywords(context)

        # 2. 搜索规则
        rules = await self.memory.search(
            query=" ".join(keywords),
            memory_type=MemoryType.RULE,
        )

        # 3. 按优先级和相关性排序
        return sorted(rules,
            key=lambda r: r.conflevance * r.metadata["confidence"],
            reverse=True
        )

    async def add_rule(self, rule_type: str, logic: str, confidence: float, example: dict):
        """添加新规则"""
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"

        await self.memory.remember(
            content=logic,
            memory_type=MemoryType.RULE,
            metadata={
                "rule_id": rule_id,
                "rule_type": rule_type,
                "confidence": confidence,
                "source": "remote_feedback",
                "example": example,
                "usage_count": 0,
                "success_count": 0,
            }
        )

    async def update_rule_stats(self, rule_id: str, success: bool):
        """更新规则统计"""
        rule = await self.memory.get_rule(rule_id)
        rule.metadata["usage_count"] += 1
        if success:
            rule.metadata["success_count"] += 1
        await self.memory.update_rule(rule)
```

### 5. RemoteAnalyzer (远程分析器)

```python
class RemoteAnalyzer:
    """远程 LLM 分析器"""

    def __init__(self, remote_backend, sanitizer):
        self.remote = remote_backend
        self.sanitizer = sanitizer

    async def analyze(self, context: dict) -> dict:
        """分析本地处理结果"""

        prompt = f"""分析以下对话处理是否正确。

## 用户输入 (已脱敏)
{context["user_input"]}

## 对话历史 (已脱敏)
{context["history"]}

## 本地回答
{context["local_answer"]}

## 本地置信度
{context["confidence"]}

## 本地推理
{context["reasoning"]}

请分析:
1. 本地回答是否正确?
2. 如果不正确，正确的回答是什么?
3. 本地的处理逻辑有什么问题?
4. 给出一个更好的处理规则

返回 JSON:
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "你的分析",
  "better_answer": "如果本地错了，正确答案是什么",
  "logic_rule": "可以学习的通用规则"
}}"""

        response = await self.remote.generate_async(prompt)
        return json.loads(response)
```

## Privacy Model

### 数据分类

| 数据类型 | 本地存储 | 发送给远程 |
|---------|---------|-----------|
| 用户原始输入 | ✅ | ❌ |
| 脱敏后输入 | ✅ | ✅ |
| 对话历史 | ✅ 原始 | ✅ 脱敏版 |
| 实体名称 | ✅ | ❌ |
| 实体类型 | ✅ | ✅ |
| 记忆内容 | ✅ | ❌ |
| 规则 | ✅ | ❌ |
| 本地回答 | ✅ | ✅ |
| 置信度 | ✅ | ✅ |

### 脱敏策略

```python
class SanitizationStrategy:
    # 永不发送
    NEVER_SEND = ["EMAIL", "PHONE", "ID_CARD", "BANK_CARD", "PASSWORD"]

    # 脱敏后发送
    SANITIZE = ["PERSON", "ADDRESS", "COMPANY"]

    # 可以发送
    ALLOW = ["TIME", "INTENT", "TOPIC", "CONFIDENCE"]
```

## Learning Mechanism

### 触发条件

```python
def should_learn(result: LocalResult) -> bool:
    return (
        result.confidence < 0.9 or      # 置信度低
        result.has_ambiguity or          # 有歧义
        result.entity_count > 2 or      # 实体多
        result.time_expression          # 有时间表达
    )
```

### 学习预算

```python
class LearningBudget:
    hourly_limit = 10
    daily_limit = 100
    min_confidence_threshold = 0.7  # 只学习置信度 < 0.7 的情况
```

### 规则生命周期

```python
class RuleLifecycle:
    # 规则过期
    expire_after_days = 90
    expire_after_unused_days = 30

    # 规则更新
    update_on_success = True
    update_on_failure = True

    # 规则删除
    delete_on_low_success_rate = True
    min_success_rate = 0.3
```

## Configuration

```yaml
# config.yaml
hybrid_agent:
  # 本地模型配置
  local:
    default_model: minicpm-v-4.6

  # 远程模型配置
  remote:
    enabled: true
    model: mimo-v2.5-pro
    api_key_env: MIMO_API_KEY

  # 学习配置
  learning:
    enabled: true
    confidence_threshold: 0.9
    hourly_limit: 10
    daily_limit: 100

  # 规则配置
  rules:
    expire_days: 90
    min_success_rate: 0.3
    max_rules: 1000

  # 隐私配置
  privacy:
    sanitize_before_remote: true
    allowed_remote_data:
      - time_expressions
      - intent
      - confidence
    blocked_remote_data:
      - person_names
      - email_addresses
      - phone_numbers
```

## Testing Strategy

1. **Unit Tests**: 各组件独立测试
2. **Integration Tests**: 完整流程测试
3. **Privacy Tests**: 确保脱敏正确
4. **Learning Tests**: 验证规则学习和应用
5. **Performance Tests**: 响应时间和资源占用

## Implementation Phases

### Phase 1: 基础框架
- [ ] RuleManager 实现
- [ ] LocalProcessor 集成规则查询
- [ ] 基础置信度评估

### Phase 2: 远程学习
- [ ] RemoteAnalyzer 实现
- [ ] LearningController 实现
- [ ] 脱敏打包流程

### Phase 3: 规则系统
- [ ] 规则优先级
- [ ] 规则冲突解决
- [ ] 规则生命周期

### Phase 4: 优化
- [ ] 学习预算控制
- [ ] 规则效果评估
- [ ] 性能优化
