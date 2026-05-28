"""Data sanitizer - 敏感数据检测和脱敏

简单方案：基于正则和关键词检测
未来可升级到 Presidio
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SensitivityLevel(Enum):
    """敏感度级别"""
    PUBLIC = 0      # 可发送给 API
    PERSONAL = 1    # 脱敏后发送
    SENSITIVE = 2   # 脱敏后发送或本地
    CRITICAL = 3    # 只用本地


@dataclass
class DetectionResult:
    """检测结果"""
    level: SensitivityLevel
    matches: list[tuple[str, str, str]]  # (类型, 原文, 占位符)
    reasons: list[str]


@dataclass
class SanitizedText:
    """脱敏后的文本"""
    original: str
    sanitized: str
    mapping: dict[str, str]  # 占位符 → 原文
    level: SensitivityLevel


class SensitivityDetector:
    """敏感度检测器"""

    # 正则模式：(pattern, sensitivity_level)
    # 注意：使用 [a-zA-Z0-9] 代替 \w，避免匹配中文字符
    REGEX_PATTERNS: dict[str, tuple[str, SensitivityLevel]] = {
        "EMAIL": (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', SensitivityLevel.PERSONAL),
        "PHONE": (r'1[3-9]\d{9}', SensitivityLevel.PERSONAL),
        "ID_CARD": (r'(?<!\d)\d{17}[\dXx](?!\d)', SensitivityLevel.CRITICAL),
        "BANK_CARD": (r'(?<!\d)\d{16,19}(?!\d)', SensitivityLevel.CRITICAL),
        "IP_ADDR": (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', SensitivityLevel.PERSONAL),
    }

    # 关键词分类
    KEYWORD_CATEGORIES: dict[str, dict[str, Any]] = {
        "health": {
            "keywords": ["医院", "生病", "吃药", "体检", "血压", "血糖",
                        "心脏病", "糖尿病", "癌症", "抑郁", "症状"],
            "level": SensitivityLevel.SENSITIVE,
        },
        "finance": {
            "keywords": ["工资", "存款", "投资", "股票", "基金", "贷款",
                        "信用卡", "支付宝", "微信支付", "余额", "转账"],
            "level": SensitivityLevel.SENSITIVE,
        },
        "password": {
            "keywords": ["密码", "口令", "验证码", "token", "secret", "密钥"],
            "level": SensitivityLevel.CRITICAL,
        },
        "work_secret": {
            "keywords": ["报价", "成本", "利润", "客户名单", "合同金额", "底价"],
            "level": SensitivityLevel.SENSITIVE,
        },
    }

    def __init__(self, custom_keywords: dict[str, dict[str, Any]] | None = None):
        # 用户标记的实体
        self.user_marked: dict[str, SensitivityLevel] = {}

        # 合并自定义关键词
        if custom_keywords:
            self.KEYWORD_CATEGORIES.update(custom_keywords)

    def detect(self, text: str) -> DetectionResult:
        """检测文本敏感度"""
        max_level = SensitivityLevel.PUBLIC
        matches = []
        reasons = []

        # 第1层：正则检测
        for name, (pattern, level) in self.REGEX_PATTERNS.items():
            for match in re.finditer(pattern, text):
                max_level = max(max_level, level, key=lambda x: x.value)
                idx = len([m for m in matches if m[0] == name])
                placeholder = f"[{name}_{idx}]"
                matches.append((name, match.group(), placeholder))
                reasons.append(f"检测到{name}: {match.group()[:15]}...")

        # 第2层：关键词检测
        for category, config in self.KEYWORD_CATEGORIES.items():
            for keyword in config["keywords"]:
                if keyword in text:
                    max_level = max(max_level, config["level"], key=lambda x: x.value)
                    reasons.append(f"检测到{category}关键词: {keyword}")

        # 第3层：用户标记
        for entity, level in self.user_marked.items():
            if entity in text:
                max_level = max(max_level, level, key=lambda x: x.value)
                idx = len([m for m in matches if m[0] == "MARKED"])
                placeholder = f"[MARKED_{idx}]"
                matches.append(("MARKED", entity, placeholder))
                reasons.append(f"用户标记的敏感实体: {entity}")

        return DetectionResult(
            level=max_level,
            matches=matches,
            reasons=reasons,
        )

    def mark_entity(self, entity: str, level: SensitivityLevel) -> None:
        """用户标记实体为敏感"""
        self.user_marked[entity] = level

    def unmark_entity(self, entity: str) -> None:
        """取消用户标记"""
        self.user_marked.pop(entity, None)

    def get_marked_entities(self) -> dict[str, SensitivityLevel]:
        """获取所有用户标记的实体"""
        return dict(self.user_marked)


class DataSanitizer:
    """数据脱敏器"""

    def __init__(self, detector: SensitivityDetector | None = None):
        self.detector = detector or SensitivityDetector()

    def sanitize(self, text: str) -> SanitizedText:
        """脱敏文本"""
        result = self.detector.detect(text)

        # 按长度倒序替换（避免短词影响长词）
        sorted_matches = sorted(result.matches, key=lambda x: len(x[1]), reverse=True)

        sanitized = text
        mapping = {}

        for entity_type, original, placeholder in sorted_matches:
            if original in sanitized:
                sanitized = sanitized.replace(original, placeholder, 1)
                mapping[placeholder] = original

        return SanitizedText(
            original=text,
            sanitized=sanitized,
            mapping=mapping,
            level=result.level,
        )

    def desanitize(self, response: str, mapping: dict[str, str]) -> str:
        """还原响应中的敏感信息"""
        result = response
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result

    def desanitize_partial(self, response: str, mapping: dict[str, str]) -> str:
        """部分还原（只还原人名，不还原邮箱电话等）"""
        name_mapping = {
            k: v for k, v in mapping.items()
            if k.startswith("[MARKED_") or k.startswith("[PERSON_")
        }
        result = response
        for placeholder, original in name_mapping.items():
            result = result.replace(placeholder, original)
        return result
