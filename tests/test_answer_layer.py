"""回答生成层测试：PromptBuilder + SafetyGuard。"""

import pytest
from medrag.rag.prompt_builder import PromptBuilder
from medrag.rag.safety_guard import SafetyGuard, RETRIEVAL_DISCLAIMERS


# ============================================================================
# PromptBuilder
# ============================================================================


class TestPromptBuilderQueryType:
    """验证不同 query_type 注入正确的分类指令。"""

    @pytest.fixture
    def builder(self):
        return PromptBuilder()

    def test_default_query_type_when_no_route(self, builder):
        """不传 route 时默认 general_medical_qa。"""
        prompt = builder.build_answer_prompt(query="感冒了怎么办")
        assert "综合医疗问答" in prompt
        assert "不使用固定五段式" in prompt

    def test_disease_fact_injects_hint(self, builder):
        prompt = builder.build_answer_prompt(
            query="糖尿病的病因是什么",
            route={"query_type": "disease_fact", "use_kg": True, "use_toyhom_qa": True},
        )
        assert "事实短答" in prompt
        assert "1-3 个短段落" in prompt

    def test_medication_injects_hint(self, builder):
        prompt = builder.build_answer_prompt(
            query="阿莫西林的剂量是多少",
            route={"query_type": "medication", "use_kg": True, "use_toyhom_qa": True},
        )
        assert "用药安全" in prompt
        assert "剂量信息必须带单位" in prompt

    def test_symptom_consult_injects_hint(self, builder):
        prompt = builder.build_answer_prompt(
            query="头痛发热是什么病",
            route={"query_type": "symptom_consult", "use_kg": True, "use_toyhom_qa": True},
        )
        assert "症状鉴别" in prompt
        assert "可能原因排序" in prompt

    def test_test_report_injects_hint(self, builder):
        prompt = builder.build_answer_prompt(
            query="需要做什么检查",
            route={"query_type": "test_report", "use_kg": True, "use_toyhom_qa": True},
        )
        assert "检查/报告解读" in prompt

    def test_diet_injects_hint(self, builder):
        prompt = builder.build_answer_prompt(
            query="糖尿病不能吃什么",
            route={"query_type": "diet", "use_kg": True, "use_toyhom_qa": True},
        )
        assert "饮食建议" in prompt

    def test_department_injects_hint(self, builder):
        prompt = builder.build_answer_prompt(
            query="感冒挂什么科",
            route={"query_type": "department", "use_kg": True, "use_toyhom_qa": True},
        )
        assert "科室咨询" in prompt

    def test_general_medical_qa_injects_hint(self, builder):
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            route={"query_type": "general_medical_qa", "use_kg": False, "use_toyhom_qa": True},
        )
        assert "综合医疗问答" in prompt
        assert "自然组织" in prompt

    def test_unknown_query_type_falls_back(self, builder):
        """未知 query_type 回退到 general_medical_qa。"""
        prompt = builder.build_answer_prompt(
            query="test",
            route={"query_type": "bogus_type", "use_kg": False, "use_toyhom_qa": True},
        )
        assert "综合医疗问答" in prompt


class TestPromptBuilderRetrievalQuality:
    """验证检索质量对置信度指令注入的影响。"""

    @pytest.fixture
    def builder(self):
        return PromptBuilder()

    def test_no_retrieval_quality_no_note(self, builder):
        """不传 retrieval_quality 时不注入置信度标记。"""
        prompt = builder.build_answer_prompt(query="感冒了怎么办")
        assert "知识库中未检索到相关资料" not in prompt

    def test_high_confidence_no_note(self, builder):
        """高置信度时走 Tier 1 完整回答（有引用要求）。"""
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            retrieval_quality={"has_kg": True, "has_qa": False, "confidence": "high"},
        )
        assert "如 [KG-1]、[QA-2]、[CASE-1]" in prompt

    def test_none_confidence_injects_note(self, builder):
        """空检索时走 Tier 2 安全回答。"""
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            retrieval_quality={"has_kg": False, "has_qa": False, "confidence": "none"},
        )
        assert "未找到确切信息" in prompt
        assert "建议咨询专业医生" in prompt

    def test_low_confidence_injects_note(self, builder):
        """低置信度时走 Tier 2 安全回答。"""
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            retrieval_quality={"has_kg": True, "has_qa": False, "confidence": "low"},
        )
        assert "未找到确切信息" in prompt
        assert "建议咨询专业医生" in prompt


class TestPromptBuilderContextAssembly:
    """验证上下文组装（非回归测试）。"""

    @pytest.fixture
    def builder(self):
        return PromptBuilder()

    def test_query_appears_in_prompt(self, builder):
        prompt = builder.build_answer_prompt(query="感冒了怎么办")
        assert "感冒了怎么办" in prompt

    def test_case_context_included(self, builder):
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办", case_context="患者有高血压病史"
        )
        assert "患者有高血压病史" in prompt
        assert "病例已有信息与通用建议" in prompt

    def test_case_results_included(self, builder):
        prompt = builder.build_answer_prompt(
            query="我的血糖高吗",
            case_results=[{"filename": "case.pdf", "answer": "空腹血糖 7.2 mmol/L"}],
        )
        assert "用户病例片段" in prompt
        assert "空腹血糖 7.2 mmol/L" in prompt

    def test_query_info_included(self, builder):
        prompt = builder.build_answer_prompt(
            query="嗓子疼怎么办",
            query_info={
                "normalized_query": "咽痛怎么办",
                "rewrite_reason": "嗓子疼->咽痛",
                "medical_terms": ["咽痛"],
            },
        )
        assert "原始问题：嗓子疼怎么办" in prompt
        assert "检索规范化问题：咽痛怎么办" in prompt

    def test_kg_results_included(self, builder):
        kg = [{"intent": "疾病简介", "answer": "感冒是上呼吸道感染。"}]
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办", kg_results=kg
        )
        assert "上呼吸道感染" in prompt

    def test_toyhom_results_answer_included(self, builder):
        qa = [{"title": "感冒了吃啥药", "answer": "建议多喝水、休息。", "department": "全科"}]
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办", toyhom_results=qa
        )
        assert "多喝水" in prompt
        assert "全科" in prompt

    def test_toyhom_results_title_fallback(self, builder):
        """当 answer 为空时，使用 title 作为文本。"""
        qa = [{"title": "感冒了吃啥药", "answer": "", "department": "全科"}]
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办", toyhom_results=qa
        )
        assert "感冒了吃啥药" in prompt

    def test_kg_truncation(self, builder):
        """验证 KG 结果截断到 MAX_PER_SOURCE(5)。"""
        kg = [{"intent": f"intent_{i}", "answer": f"answer_{i}"} for i in range(10)]
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办", kg_results=kg
        )
        # [6] 不应出现
        assert "[6]" not in prompt

    def test_long_result_truncation(self, builder):
        """验证单条结果截断到 MAX_RESULT_CHARS(400)。"""
        long_text = "X" * 500
        kg = [{"intent": "test", "answer": long_text}]
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办", kg_results=kg
        )
        assert "…" in prompt
        assert "XXXXX" in prompt


# ============================================================================
# SafetyGuard
# ============================================================================


class TestSafetyGuardRiskDetection:
    """验证风险分级检测。"""

    @pytest.fixture
    def guard(self):
        return SafetyGuard()

    def test_no_risk_for_safe_query(self, guard):
        result = guard.detect_risk("感冒了怎么办")
        assert result["is_high_risk"] is False
        assert result["is_moderate_risk"] is False
        assert result["risk_level"] == "none"
        assert result["safety_message"] == ""

    # ---- 红色信号 ----

    @pytest.mark.parametrize("keyword", ["胸痛", "呼吸困难", "意识不清", "抽搐", "大出血", "休克", "过量服药", "自杀"])
    def test_red_signal_detected(self, guard, keyword):
        result = guard.detect_risk(f"我有{keyword}的症状")
        assert result["is_high_risk"] is True
        assert result["risk_level"] == "red"
        assert "120" in result["safety_message"] or "急诊" in result["safety_message"]

    # ---- 黄色信号 ----

    @pytest.mark.parametrize("keyword", ["便血", "黑便", "高热不退", "剧烈腹痛", "孕妇", "婴儿"])
    def test_yellow_signal_detected(self, guard, keyword):
        result = guard.detect_risk(f"我有{keyword}的情况")
        assert result["is_moderate_risk"] is True
        assert result["risk_level"] == "yellow"
        assert "尽快就医" in result["safety_message"]

    # ---- 红色优先于黄色 ----

    def test_red_overrides_yellow_for_shared_keyword(self, guard):
        """呕血同时出现在红/黄信号中，应以红色为准。"""
        result = guard.detect_risk("呕血了怎么办")
        assert result["is_high_risk"] is True
        assert result["risk_level"] == "red"

    def test_red_plus_yellow_yields_red_only(self, guard):
        result = guard.detect_risk("胸痛且便血，怎么办")
        assert result["is_high_risk"] is True
        assert result["risk_level"] == "red"
        assert "胸痛" in result["risk_types"]


class TestSafetyGuardDisclaimers:
    """验证检索质量免责声明。"""

    @pytest.fixture
    def guard(self):
        return SafetyGuard()

    def test_high_disclaimer(self, guard):
        msg = guard.get_retrieval_disclaimer("high")
        assert "请咨询医生确认后使用" in msg

    def test_low_disclaimer(self, guard):
        msg = guard.get_retrieval_disclaimer("low")
        assert "差异" in msg or "不完整" in msg

    def test_none_disclaimer(self, guard):
        msg = guard.get_retrieval_disclaimer("none")
        assert "未检索到" in msg

    def test_unknown_quality_falls_back(self, guard):
        """未知检索质量回退到 high 级别的免责声明。"""
        msg = guard.get_retrieval_disclaimer("bogus")
        assert len(msg) > 0


class TestSafetyGuardAppendNotice:
    """验证安全提示注入位置。"""

    @pytest.fixture
    def guard(self):
        return SafetyGuard()

    def test_red_inserted_at_beginning(self, guard):
        risk = guard.detect_risk("我胸痛了")
        result = guard.append_safety_notice("核心结论：请就医。", risk)
        # 红色警告应在回答之前
        assert result.index("120") < result.index("核心结论")

    def test_yellow_appended_at_end(self, guard):
        risk = guard.detect_risk("我有便血的情况")
        result = guard.append_safety_notice("核心结论：观察。", risk)
        # 黄色警告应在回答之后
        assert result.index("尽快就医") > result.index("核心结论")

    def test_retrieval_disclaimer_applied(self, guard):
        risk = guard.detect_risk("感冒了怎么办")
        result = guard.append_safety_notice("核心结论。", risk, retrieval_quality="none")
        assert "未检索到" in result

    def test_no_risk_no_warning(self, guard):
        risk = guard.detect_risk("感冒了怎么办")
        result = guard.append_safety_notice("核心结论。", risk, retrieval_quality="high")
        assert "120" not in result
        assert "尽快就医" not in result
