"""Tests for Presidio-based sanitizer."""

import pytest

# 跳过 Presidio 未安装的情况
presidio = pytest.importorskip("presidio_analyzer")

from server.sanitizer_presidio import (
    PresidioDetector,
    PresidioSanitizer,
    create_sanitizer,
)
from server.sanitizer import SensitivityLevel


class TestPresidioDetector:
    """Presidio 检测器测试"""

    def test_detect_person(self):
        detector = PresidioDetector(language="en")
        result = detector.detect("John Smith said he would come")
        # Presidio 应该能识别出人名
        assert any(m[0] == "PERSON" for m in result.matches)

    def test_detect_email(self):
        detector = PresidioDetector(language="en")
        result = detector.detect("Contact me at john@example.com")
        assert any(m[0] == "EMAIL_ADDRESS" for m in result.matches)
        assert result.level in [SensitivityLevel.PERSONAL, SensitivityLevel.SENSITIVE]

    def test_detect_phone(self):
        detector = PresidioDetector(language="en")
        result = detector.detect("Call me at 13800138000")
        assert any(m[0] == "PHONE_NUMBER" for m in result.matches)

    def test_detect_credit_card(self):
        detector = PresidioDetector(language="en")
        result = detector.detect("Card number 4111111111111111")
        assert any(m[0] == "CREDIT_CARD" for m in result.matches)
        assert result.level == SensitivityLevel.CRITICAL

    def test_detect_multiple(self):
        detector = PresidioDetector(language="en")
        result = detector.detect("John's email is john@test.com, phone 13800138000")
        assert len(result.matches) >= 2

    def test_custom_keywords_still_work(self):
        custom = {
            "project": {
                "keywords": ["ProjectX"],
                "level": SensitivityLevel.SENSITIVE,
            }
        }
        detector = PresidioDetector(language="en", custom_keywords=custom)
        result = detector.detect("Let's discuss ProjectX")
        assert result.level == SensitivityLevel.SENSITIVE


class TestPresidioSanitizer:
    """Presidio 脱敏器测试"""

    def test_sanitize_presidio(self):
        sanitizer = PresidioSanitizer(language="en")
        result = sanitizer.sanitize("John's email is john@example.com")
        # 应该脱敏了邮箱
        assert "[EMAIL_ADDRESS_0]" in result.sanitized or "[EMAIL_0]" in result.sanitized

    def test_sanitize_with_presidio_native(self):
        sanitizer = PresidioSanitizer(language="en")
        result = sanitizer.sanitize_with_presidio("John lives in New York")
        # Presidio 原生脱敏应该工作
        assert result.sanitized != result.original


class TestCreateSanitizer:
    """工厂函数测试"""

    def test_create_simple(self):
        sanitizer = create_sanitizer(strategy="simple")
        assert sanitizer.__class__.__name__ == "DataSanitizer"

    def test_create_presidio(self):
        try:
            sanitizer = create_sanitizer(strategy="presidio", language="en")
            assert sanitizer.__class__.__name__ == "PresidioSanitizer"
        except ImportError:
            pytest.skip("Presidio not installed")

    def test_create_auto(self):
        sanitizer = create_sanitizer(strategy="auto")
        # 应该创建一个可用的 sanitizer
        result = sanitizer.sanitize("test@example.com")
        assert "[EMAIL" in result.sanitized
