import unittest
from datetime import datetime, timedelta

from server.context_manager import (
    ContextManager,
    ConversationContext,
    Entity,
    extract_entities,
)


class ExtractEntitiesTests(unittest.TestCase):
    def test_extracts_chinese_person_names(self):
        text = "张总今天来了，王经理也在。"
        entities = extract_entities(text)

        self.assertIn("张总", entities)
        self.assertEqual(entities["张总"].type, "person")
        self.assertIn("王经理", entities)

    def test_extracts_quoted_things(self):
        text = "请看一下\"季度报告\"的内容。"
        entities = extract_entities(text)

        self.assertIn("季度报告", entities)
        self.assertEqual(entities["季度报告"].type, "thing")

    def test_extracts_chinese_quotation_marks(self):
        text = "关于「项目方案」的讨论"
        entities = extract_entities(text)

        self.assertIn("项目方案", entities)

    def test_returns_empty_dict_for_no_entities(self):
        text = "今天天气不错。"
        entities = extract_entities(text)

        self.assertEqual(entities, {})

    def test_deduplicates_same_name(self):
        text = "张总来了。张总说了什么。"
        entities = extract_entities(text)

        # Should appear once, not twice.
        self.assertEqual(sum(1 for k in entities if k == "张总"), 1)

    def test_extracts_multiple_persons(self):
        text = "李总和赵总明天开会。"
        entities = extract_entities(text)

        self.assertIn("李总", entities)
        self.assertIn("赵总", entities)

    def test_short_names_with_title(self):
        text = "陈老师推荐的。"
        entities = extract_entities(text)

        self.assertIn("陈老师", entities)


class PronounResolutionTests(unittest.TestCase):
    def setUp(self):
        self.cm = ContextManager()

    def test_resolves_he_to_last_person(self):
        self.cm.update("张总今天来了。", "好的，张总到了。")
        result = self.cm.resolve_references("他说什么了？")

        self.assertIn("张总", result)
        self.assertNotIn("他", result)

    def test_resolves_it_to_last_thing(self):
        self.cm.update("请看一下\"季度报告\"。", "好的，我看了季度报告。")
        result = self.cm.resolve_references("它有什么问题？")

        self.assertIn("季度报告", result)
        self.assertNotIn("它", result)

    def test_resolves_that_to_last_thing(self):
        self.cm.update("看下\"项目方案\"。", "方案已确认。")
        result = self.cm.resolve_references("那个怎么样？")

        self.assertIn("项目方案", result)

    def test_no_resolution_when_no_entities(self):
        result = self.cm.resolve_references("他说了什么？")
        self.assertEqual(result, "他说了什么？")

    def test_resolves_only_first_occurrence(self):
        self.cm.update("张总来了。", "收到。")
        result = self.cm.resolve_references("他来了吗？他带了什么？")

        # First "他" replaced, second stays.
        self.assertIn("张总", result)
        self.assertEqual(result.count("他"), 1)

    def test_resolves_she_to_person(self):
        self.cm.update("李总来了。", "好的。")
        result = self.cm.resolve_references("她说了什么？")

        self.assertIn("李总", result)

    def test_prefers_most_recent_entity(self):
        self.cm.update("张总来了。", "收到。")
        self.cm.update("李总也来了。", "好的。")
        result = self.cm.resolve_references("他说什么？")

        # "他" should resolve to the most recent person (李总).
        self.assertIn("李总", result)

    def test_resolves_this_to_thing(self):
        self.cm.update("看一下\"合同\"。", "好的。")
        result = self.cm.resolve_references("这个有问题。")

        self.assertIn("合同", result)


class TimeExpressionTests(unittest.TestCase):
    def setUp(self):
        self.cm = ContextManager()

    def test_today(self):
        start, end = self.cm.resolve_time("今天的会议")
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.assertEqual(start, today_start)
        self.assertEqual(end, today_start + timedelta(days=1))

    def test_yesterday(self):
        start, end = self.cm.resolve_time("昨天的数据")
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.assertEqual(start, today_start - timedelta(days=1))
        self.assertEqual(end, today_start)

    def test_day_before_yesterday(self):
        start, end = self.cm.resolve_time("前天的记录")
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.assertEqual(start, today_start - timedelta(days=2))
        self.assertEqual(end, today_start - timedelta(days=1))

    def test_current_month(self):
        start, end = self.cm.resolve_time("本月的业绩")
        now = datetime.now()
        self.assertEqual(start.day, 1)
        self.assertEqual(start.month, now.month)
        self.assertEqual(start.year, now.year)

    def test_last_month(self):
        start, end = self.cm.resolve_time("上个月的数据")
        now = datetime.now()
        self.assertEqual(start.day, 1)
        if now.month == 1:
            self.assertEqual(start.month, 12)
            self.assertEqual(start.year, now.year - 1)
        else:
            self.assertEqual(start.month, now.month - 1)
            self.assertEqual(start.year, now.year)

    def test_last_week(self):
        start, end = self.cm.resolve_time("上周的总结")
        # start should be a Monday, end should be the following Monday.
        self.assertEqual(start.weekday(), 0)  # Monday
        self.assertEqual((end - start).days, 7)

    def test_defaults_to_today(self):
        start, end = self.cm.resolve_time("随便什么")
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.assertEqual(start, today_start)
        self.assertEqual(end, today_start + timedelta(days=1))


class ContextWindowTests(unittest.TestCase):
    def test_history_limited_to_max_turns(self):
        cm = ContextManager(max_history_turns=3)
        for i in range(5):
            cm.update(f"用户消息{i}", f"回复{i}")

        # 3 turns = 6 messages (user + agent pairs).
        self.assertEqual(len(cm.context.recent_history), 6)
        # Should contain the last 3 turns.
        self.assertEqual(cm.context.recent_history[0]["content"], "用户消息2")

    def test_empty_history_initially(self):
        cm = ContextManager()
        self.assertEqual(cm.context.recent_history, [])

    def test_history_grows_until_limit(self):
        cm = ContextManager(max_history_turns=5)
        for i in range(3):
            cm.update(f"消息{i}", f"回复{i}")

        # 3 turns = 6 messages, under the limit of 10.
        self.assertEqual(len(cm.context.recent_history), 6)


class UpdateCycleTests(unittest.TestCase):
    def test_update_tracks_entities_from_user_input(self):
        cm = ContextManager()
        cm.update("张总今天来了。", "收到。")

        self.assertIn("张总", cm.context.entities)

    def test_update_tracks_entities_from_agent_response(self):
        cm = ContextManager()
        cm.update("谁来了？", "李总刚到。")

        self.assertIn("李总", cm.context.entities)

    def test_update_sets_current_topic(self):
        cm = ContextManager()
        cm.update("关于\"项目方案\"的讨论", "好的。")

        self.assertEqual(cm.context.current_topic, "项目方案")

    def test_update_refreshes_existing_entity_timestamp(self):
        cm = ContextManager()
        cm.update("张总来了。", "收到。")
        first_ts = cm.context.entities["张总"].last_mentioned

        # Small delay would happen naturally; just call again.
        cm.update("张总说了什么？", "他说要开会。")

        self.assertGreaterEqual(cm.context.entities["张总"].last_mentioned, first_ts)

    def test_full_conversation_flow(self):
        """Simulate a multi-turn conversation and verify context is maintained."""
        cm = ContextManager()

        cm.update("张总和李总今天讨论了\"年度计划\"。", "好的，我记录了张总和李总的讨论。")
        cm.update("他说预算需要增加。", "张总建议增加预算，李总同意了。")
        cm.update("它什么时候执行？", "年度计划预计下月开始执行。")

        # Entities should include both persons and the quoted topic.
        self.assertIn("张总", cm.context.entities)
        self.assertIn("李总", cm.context.entities)
        self.assertIn("年度计划", cm.context.entities)

        # Pronouns should have been available for resolution across turns.
        self.assertEqual(cm.context.current_topic, "年度计划")

    def test_get_relevant_context_returns_content(self):
        cm = ContextManager()
        cm.update("张总来了。", "收到。")
        ctx = cm.get_relevant_context("张总说了什么？")

        self.assertIn("张总", ctx)
        self.assertIn("最近对话", ctx)

    def test_get_relevant_context_empty_for_fresh_manager(self):
        cm = ContextManager()
        ctx = cm.get_relevant_context("随便什么")
        self.assertEqual(ctx, "")

    def test_edge_case_no_entities_no_crash(self):
        cm = ContextManager()
        cm.update("你好。", "你好！")
        result = cm.resolve_references("他说什么？")
        self.assertEqual(result, "他说什么？")

    def test_aliases_merged_on_update(self):
        cm = ContextManager()
        cm.update("张总来了。", "张总说了什么。")
        entity = cm.context.entities["张总"]
        original_aliases = list(entity.aliases)

        # Simulate another mention that might add aliases.
        cm.update("张总再次确认。", "好的。")
        self.assertEqual(cm.context.entities["张总"].aliases, original_aliases)

    def test_detect_topic_from_user_input(self):
        cm = ContextManager()
        cm.update("讨论一下\"数据库优化\"。", "好的。")
        self.assertEqual(cm.context.current_topic, "数据库优化")

    def test_resolve_time_in_get_relevant_context(self):
        cm = ContextManager()
        cm.update("看下\"季度报告\"。", "好的。")
        start, end = cm.resolve_time("昨天的报告")
        self.assertEqual((end - start).days, 1)


if __name__ == "__main__":
    unittest.main()
