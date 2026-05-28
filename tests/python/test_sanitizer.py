"""Tests for data sanitizer."""

import pytest
from server.sanitizer import (
    SensitivityDetector,
    SensitivityLevel,
    DataSanitizer,
)


class TestSensitivityDetector:
    """敏感度检测器测试"""

    def test_detect_email(self):
        detector = SensitivityDetector()
        result = detector.detect("我的邮箱是test@example.com")
        assert result.level == SensitivityLevel.PERSONAL
        assert len(result.matches) == 1
        assert result.matches[0][0] == "EMAIL"
        assert result.matches[0][1] == "test@example.com"

    def test_detect_phone(self):
        detector = SensitivityDetector()
        result = detector.detect("电话是13800138000")
        assert result.level == SensitivityLevel.PERSONAL
        assert len(result.matches) == 1
        assert result.matches[0][0] == "PHONE"

    def test_detect_id_card(self):
        detector = SensitivityDetector()
        result = detector.detect("身份证110101199001011234")
        assert result.level == SensitivityLevel.CRITICAL
        # ID_CARD 会匹配，BANK_CARD 也会匹配（都是数字）
        assert any(m[0] == "ID_CARD" for m in result.matches)

    def test_detect_bank_card(self):
        detector = SensitivityDetector()
        result = detector.detect("银行卡6222021234567890123")
        assert result.level == SensitivityLevel.CRITICAL

    def test_detect_multiple(self):
        detector = SensitivityDetector()
        result = detector.detect("邮箱test@example.com，电话13800138000")
        assert len(result.matches) == 2
        assert result.level == SensitivityLevel.PERSONAL

    def test_detect_health_keyword(self):
        detector = SensitivityDetector()
        result = detector.detect("我昨天去医院体检了")
        assert result.level == SensitivityLevel.SENSITIVE
        assert any("health" in r for r in result.reasons)

    def test_detect_finance_keyword(self):
        detector = SensitivityDetector()
        result = detector.detect("这个月工资发了多少")
        assert result.level == SensitivityLevel.SENSITIVE

    def test_detect_password_keyword(self):
        detector = SensitivityDetector()
        result = detector.detect("密码是什么")
        assert result.level == SensitivityLevel.CRITICAL

    def test_detect_user_marked(self):
        detector = SensitivityDetector()
        detector.mark_entity("张总", SensitivityLevel.SENSITIVE)
        result = detector.detect("张总说下周要开会")
        assert result.level == SensitivityLevel.SENSITIVE
        assert any("MARKED" in m[0] for m in result.matches)

    def test_detect_public_text(self):
        detector = SensitivityDetector()
        result = detector.detect("今天天气怎么样")
        assert result.level == SensitivityLevel.PUBLIC
        assert len(result.matches) == 0

    def test_mark_unmark_entity(self):
        detector = SensitivityDetector()
        detector.mark_entity("张总", SensitivityLevel.SENSITIVE)
        assert "张总" in detector.get_marked_entities()

        detector.unmark_entity("张总")
        assert "张总" not in detector.get_marked_entities()

    def test_custom_keywords(self):
        custom = {
            "project": {
                "keywords": ["项目A", "项目B"],
                "level": SensitivityLevel.SENSITIVE,
            }
        }
        detector = SensitivityDetector(custom_keywords=custom)
        result = detector.detect("项目A的进度如何")
        assert result.level == SensitivityLevel.SENSITIVE


class TestDataSanitizer:
    """数据脱敏器测试"""

    def test_sanitize_email(self):
        sanitizer = DataSanitizer()
        result = sanitizer.sanitize("邮箱是test@example.com")
        assert "[EMAIL_0]" in result.sanitized
        assert result.mapping["[EMAIL_0]"] == "test@example.com"
        assert result.level == SensitivityLevel.PERSONAL

    def test_sanitize_phone(self):
        sanitizer = DataSanitizer()
        result = sanitizer.sanitize("电话13800138000")
        assert "[PHONE_0]" in result.sanitized
        assert result.mapping["[PHONE_0]"] == "13800138000"

    def test_sanitize_multiple(self):
        sanitizer = DataSanitizer()
        result = sanitizer.sanitize("张三的邮箱是zhang@test.com，电话13800138000")
        assert "[EMAIL_0]" in result.sanitized
        assert "[PHONE_0]" in result.sanitized
        assert len(result.mapping) == 2

    def test_desanitize(self):
        sanitizer = DataSanitizer()
        result = sanitizer.sanitize("邮箱是test@example.com")
        response = f"好的，已记录{result.sanitized}"
        restored = sanitizer.desanitize(response, result.mapping)
        assert "test@example.com" in restored
        assert "[EMAIL_0]" not in restored

    def test_desanitize_partial(self):
        sanitizer = DataSanitizer()
        sanitizer.detector.mark_entity("张总", SensitivityLevel.SENSITIVE)
        result = sanitizer.sanitize("张总的邮箱是zhang@test.com")

        response = f"{result.sanitized}的需求"
        restored = sanitizer.desanitize_partial(response, result.mapping)
        # 人名还原，邮箱不还原
        assert "张总" in restored
        assert "[EMAIL_0]" in restored

    def test_sanitize_preserves_order(self):
        sanitizer = DataSanitizer()
        text = "第一个邮箱a@test.com，第二个邮箱b@test.com"
        result = sanitizer.sanitize(text)
        assert "[EMAIL_0]" in result.sanitized
        assert "[EMAIL_1]" in result.sanitized
        assert result.mapping["[EMAIL_0]"] == "a@test.com"
        assert result.mapping["[EMAIL_1]"] == "b@test.com"

    def test_sanitize_no_sensitive(self):
        sanitizer = DataSanitizer()
        result = sanitizer.sanitize("今天天气怎么样")
        assert result.sanitized == "今天天气怎么样"
        assert result.level == SensitivityLevel.PUBLIC
        assert len(result.mapping) == 0
