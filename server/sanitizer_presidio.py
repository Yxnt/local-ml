"""Data sanitizer - Presidio 版本

基于 Microsoft Presidio 的高精度脱敏方案
支持 NLP 实体识别 + 自定义模式
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from server.sanitizer import (
    DataSanitizer,
    DetectionResult,
    SanitizedText,
    SensitivityDetector,
    SensitivityLevel,
)

# 检查 Presidio 是否可用
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False


class PresidioDetector(SensitivityDetector):
    """基于 Presidio 的敏感度检测器"""

    def __init__(self, language: str = "en", custom_keywords: dict[str, dict[str, Any]] | None = None):
        super().__init__(custom_keywords)

        if not HAS_PRESIDIO:
            raise ImportError("pip install presidio-analyzer presidio-anonymizer")

        self._language = language

        # 尝试加载 spaCy 模型
        try:
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": language, "model_name": self._get_spacy_model(language)}
                ],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()

            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=[language],
            )
            self._has_nlp = True
        except Exception as e:
            print(f"spaCy 模型加载失败: {e}，使用纯 Presidio 模式")
            # 使用默认配置（无 NLP，只有模式匹配）
            self._analyzer = AnalyzerEngine(supported_languages=[language])
            self._has_nlp = False

        # 添加自定义识别器
        self._add_custom_recognizers()

    def _get_spacy_model(self, language: str) -> str:
        """获取 spaCy 模型名称"""
        models = {
            "en": "en_core_web_sm",
            "zh": "zh_core_web_sm",
        }
        return models.get(language, "en_core_web_sm")

    def _add_custom_recognizers(self) -> None:
        """添加自定义识别器"""

        # 中国手机号
        phone_pattern = Pattern(
            name="chinese_phone",
            regex=r"1[3-9]\d{9}",
            score=0.8,
        )
        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[phone_pattern],
            supported_language=self._language,
            name="ChinesePhoneRecognizer",
        )
        self._analyzer.registry.add_recognizer(phone_recognizer)

        # 中国身份证号
        id_card_pattern = Pattern(
            name="chinese_id_card",
            regex=r"\d{17}[\dXx]",
            score=0.9,
        )
        id_card_recognizer = PatternRecognizer(
            supported_entity="ID_CARD",
            patterns=[id_card_pattern],
            supported_language=self._language,
            name="ChineseIdCardRecognizer",
        )
        self._analyzer.registry.add_recognizer(id_card_recognizer)

    def detect(self, text: str) -> DetectionResult:
        """使用 Presidio 检测敏感信息"""
        # Presidio 分析
        presidio_results = self._analyzer.analyze(
            text=text,
            language=self._language,
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                "LOCATION", "CREDIT_CARD", "IP_ADDRESS",
                "ID_CARD",
            ],
        )

        # 转换为我们的格式
        matches = []
        reasons = []
        max_level = SensitivityLevel.PUBLIC

        for result in presidio_results:
            entity_type = result.entity_type
            start = result.start
            end = result.end
            matched_text = text[start:end]
            score = result.score

            # 确定敏感级别
            level = self._get_sensitivity_level(entity_type, score)
            max_level = max(max_level, level, key=lambda x: x.value)

            # 生成占位符
            idx = len([m for m in matches if m[0] == entity_type])
            placeholder = f"[{entity_type}_{idx}]"

            matches.append((entity_type, matched_text, placeholder))
            reasons.append(f"Presidio 检测到 {entity_type}: {matched_text} (置信度: {score:.2f})")

        # 也运行关键词检测（Presidio 不覆盖所有场景）
        keyword_result = super().detect(text)
        for match in keyword_result.matches:
            if match[1] not in [m[1] for m in matches]:  # 避免重复
                matches.append(match)
        reasons.extend(keyword_result.reasons)
        max_level = max(max_level, keyword_result.level, key=lambda x: x.value)

        return DetectionResult(
            level=max_level,
            matches=matches,
            reasons=reasons,
        )

    def _get_sensitivity_level(self, entity_type: str, score: float) -> SensitivityLevel:
        """根据实体类型和置信度确定敏感级别"""
        critical_types = {"CREDIT_CARD", "ID_CARD"}
        sensitive_types = {"PERSON", "PHONE_NUMBER"}
        personal_types = {"EMAIL_ADDRESS", "LOCATION", "IP_ADDRESS"}

        if entity_type in critical_types:
            return SensitivityLevel.CRITICAL
        elif entity_type in sensitive_types:
            return SensitivityLevel.SENSITIVE if score > 0.7 else SensitivityLevel.PERSONAL
        elif entity_type in personal_types:
            return SensitivityLevel.PERSONAL
        else:
            return SensitivityLevel.PUBLIC


class PresidioSanitizer(DataSanitizer):
    """基于 Presidio 的脱敏器"""

    def __init__(self, language: str = "en", custom_keywords: dict[str, dict[str, Any]] | None = None):
        detector = PresidioDetector(language=language, custom_keywords=custom_keywords)
        super().__init__(detector)

        # 初始化匿名化器
        self._anonymizer = AnonymizerEngine()

    def sanitize_with_presidio(self, text: str) -> SanitizedText:
        """使用 Presidio 原生匿名化（更准确）"""
        # 分析
        analyzer_results = self._detector._analyzer.analyze(
            text=text,
            language=self._detector._language,
        )

        # 使用 Presidio 匿名化
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={
                "PERSON": OperatorConfig("replace", {"new_value": "[姓名]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[邮箱]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[电话]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[地址]"}),
                "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[银行卡]"}),
                "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[IP]"}),
                "ID_CARD": OperatorConfig("replace", {"new_value": "[身份证]"}),
            },
        )

        # 构建映射
        mapping = {}
        for result in analyzer_results:
            original = text[result.start:result.end]
            placeholder = anonymized.text[result.start:result.end]
            if placeholder != original:
                mapping[placeholder] = original

        return SanitizedText(
            original=text,
            sanitized=anonymized.text,
            mapping=mapping,
            level=self._detector.detect(text).level,
        )


def create_sanitizer(strategy: str = "auto", language: str = "en", **kwargs) -> DataSanitizer:
    """工厂函数：创建脱敏器"""
    if strategy == "presidio":
        if not HAS_PRESIDIO:
            print("Presidio 未安装，使用简单方案")
            return DataSanitizer()
        return PresidioSanitizer(language=language, **kwargs)

    elif strategy == "simple":
        return DataSanitizer()

    else:  # auto
        if HAS_PRESIDIO:
            try:
                return PresidioSanitizer(language=language, **kwargs)
            except Exception as e:
                print(f"Presidio 初始化失败: {e}，使用简单方案")
                return DataSanitizer()
        else:
            return DataSanitizer()
