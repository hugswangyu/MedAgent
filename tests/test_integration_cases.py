"""集成测试：病例上传、聊天引用、删除清理全链路。"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medrag.data.user_case_store import (
    UserCaseRetriever,
    add_user_case,
    get_user_cases,
    remove_user_case,
)


# ============================================================================
# 上传链路：解析 → 脱敏 → 分块 → 摘要 → 入库 → 向量化
# ============================================================================


class TestUploadPipeline:
    """验证 _run_upload_job 中各步骤的调用顺序和参数。"""

    @patch("medrag.data.case_parser.parse_case_file")
    @patch("medrag.data.text_cleaner.clean_medical_text")
    @patch("medrag.data.text_cleaner.desensitize_medical_text")
    @patch("medrag.app.api.documents.add_user_case")
    @patch("medrag.app.api.documents.add_document")
    @patch("medrag.vectors.milvus_client.MilvusClientWrapper")
    @patch("medrag.vectors.embedding.EmbeddingModel")
    def test_upload_full_pipeline(
        self,
        mock_embedding_model,
        mock_milvus_client,
        mock_add_document,
        mock_add_user_case,
        mock_desensitize,
        mock_clean,
        mock_parse,
    ):
        from medrag.app.api.documents import _run_upload_job, _split_text

        # Setup mocks
        mock_parse.return_value = "患者主诉：头痛一周。\n\n既往史：高血压。\n\n检查：血压 150/95。"
        mock_clean.return_value = "患者主诉：头痛一周。\n\n既往史：高血压。\n\n检查：血压 150/95。"
        mock_desensitize.return_value = mock_clean.return_value

        mock_model_instance = MagicMock()
        mock_model_instance.encode.return_value = [[0.1] * 768]
        mock_embedding_model.return_value = mock_model_instance

        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.collection = mock_collection
        mock_milvus_client.return_value = mock_client_instance

        file_bytes = b"dummy content"
        _run_upload_job(
            job_id="test-job-1",
            file_bytes=file_bytes,
            original_filename="report.txt",
            username="testuser",
        )

        # verify parsing
        mock_parse.assert_called_once()
        # verify cleaning
        mock_clean.assert_called_once()
        mock_desensitize.assert_called_once()
        # verify add_user_case was called with chunks
        mock_add_user_case.assert_called_once()
        args, kwargs = mock_add_user_case.call_args
        assert kwargs["username"] == "testuser"
        assert kwargs["filename"] == "report.txt"
        assert len(kwargs["chunks"]) > 0
        assert "summary" in kwargs
        # verify add_document was called
        mock_add_document.assert_called_once()
        args, kwargs = mock_add_document.call_args
        assert kwargs["username"] == "testuser"
        assert kwargs["document_id"] is not None
        # Private cases must never be written to the public Milvus collection.
        mock_embedding_model.assert_not_called()
        mock_milvus_client.assert_not_called()

    def test_split_text_basic(self):
        """验证基本文本分块逻辑。"""
        from medrag.app.api.documents import _split_text

        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = _split_text(text, chunk_size=500, overlap=100)
        assert len(chunks) == 3
        assert "第一段内容" in chunks[0]
        assert "第二段内容" in chunks[1]
        assert "第三段内容" in chunks[2]

    def test_split_text_chunk_overflow(self):
        """验证超长段落正确切分。"""
        from medrag.app.api.documents import _split_text

        long_para = "测试" * 300  # 600 chars > chunk_size
        text = f"{long_para}\n\n短段落。"
        chunks = _split_text(text, chunk_size=200, overlap=50)
        assert len(chunks) >= 2  # 长段落被拆分 + 短段落

    def test_build_case_summary(self):
        """验证摘要提取关键字段。"""
        from medrag.app.api.documents import _build_case_summary

        text = (
            "主诉：头痛一周\n"
            "现病史：患者1周前无明显诱因出现头痛\n"
            "既往史：高血压病史5年\n"
            "检查：血压150/95mmHg\n"
            "诊断：高血压2级\n"
            "用药：氨氯地平5mg qd\n"
            "一些无关的叙述文字\n"
        )
        summary = _build_case_summary(text)
        assert "主诉" in summary
        assert "既往史" in summary
        assert "诊断" in summary
        assert "用药" in summary


# ============================================================================
# 聊天链路：病例进入 prompt
# ============================================================================


class TestChatWithCaseContext:
    """验证聊天链路中病例上下文和片段正确注入 prompt。"""

    def test_case_context_and_results_in_prompt(self):
        """当有病例摘要和检索结果时，prompt 应包含两者。"""
        from medrag.rag.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        prompt = builder.build_answer_prompt(
            query="我的血糖高吗",
            case_context="检查结果：空腹血糖 7.2 mmol/L",
            case_results=[
                {"filename": "report.pdf", "answer": "空腹血糖 7.2 mmol/L，偏高"}
            ],
            route={"query_type": "test_report", "answer_style": "test_report"},
        )

        # 病例摘要应出现在 prompt 中
        assert "检查结果：空腹血糖 7.2 mmol/L" in prompt

        # 病例片段也应出现
        assert "用户病例片段" in prompt
        assert "空腹血糖 7.2 mmol/L" in prompt
        assert "report.pdf" in prompt

        # 当有病例时应该用 case_based 风格
        assert "病例已有信息与通用建议" in prompt

    def test_normal_query_no_case_no_reference(self):
        """普通医学问题（无病例上下文）不应引用病例。"""
        from medrag.rag.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            route={"query_type": "general_medical_qa", "answer_style": "general_guidance"},
        )

        # 不应包含病例相关标题和内容
        assert "用户病例片段" in prompt  # 这个 section 始终存在，但内容是空的
        assert "（此部分暂无可用资料）" in prompt  # 空占位

        # 不应该是 case_based 风格（没有 case_context 也没有 case_results）
        assert "病例已有信息与通用建议" not in prompt

    def test_case_results_limited_to_max(self):
        """验证病例结果被限制在 MAX_PER_SOURCE(5) 内。"""
        from medrag.rag.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        many_results = [
            {"filename": f"f{i}.pdf", "answer": f"内容{i}"}
            for i in range(10)
        ]
        prompt = builder.build_answer_prompt(
            query="我的血糖高吗",
            case_results=many_results,
            route={"query_type": "test_report", "answer_style": "test_report"},
        )

        # 最多只显示前5条病例（没有[编号]标记了）
        assert prompt.count("文件：") <= 5


# ============================================================================
# 删除链路：同步清理
# ============================================================================


class TestDeletePipeline:
    """验证删除链路中所有资源的清理。"""

    def test_delete_cleanup_all_resources(self):
        from medrag.app.api.documents import _run_delete_job
        from unittest.mock import patch as _patch

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.collection = mock_collection
        mock_isdir = MagicMock(return_value=True)
        mock_listdir = MagicMock(return_value=["job1_report.txt"])
        mock_remove = MagicMock()
        mock_remove_doc = MagicMock()
        mock_remove_case = MagicMock()

        with _patch("medrag.vectors.milvus_client.MilvusClientWrapper",
                    return_value=mock_client) as mock_milvus_wrapper, \
             _patch("medrag.app.api.documents.os.path.isdir", mock_isdir), \
             _patch("medrag.app.api.documents.os.listdir", mock_listdir), \
             _patch("medrag.app.api.documents.os.remove", mock_remove), \
             _patch("medrag.app.api.documents.remove_document", mock_remove_doc), \
             _patch("medrag.app.api.documents.remove_user_case", mock_remove_case):
            _run_delete_job("del-job-1", "report.txt", "testuser")

        # Private case deletion must never touch the public Milvus collection.
        mock_milvus_wrapper.assert_not_called()
        mock_collection.delete.assert_not_called()

        # 2. remove_document 被调用
        mock_remove_doc.assert_called_once_with("report.txt", username="testuser")

        # 3. remove_user_case 被调用
        mock_remove_case.assert_called_once_with("testuser", "report.txt")

    def test_user_case_store_remove_is_idempotent(self):
        """删除不存在的病例应返回 False 而不是报错。"""
        result = remove_user_case("nonexistent_user", "nonexistent.pdf")
        assert result is False

    def test_document_store_remove_is_idempotent(self, tmp_path, monkeypatch):
        """删除不存在的文档索引应返回 False。"""
        from medrag.app import document_store
        store_path = tmp_path / "docs_index.json"
        monkeypatch.setattr(document_store, "_doc_store",
                            document_store.JsonStore(str(store_path)))

        result = document_store.remove_document("nonexistent.pdf", username="testuser")
        assert result is False


# ============================================================================
# 端到端：病例隔离性
# ============================================================================


class TestCaseIsolation:
    """验证用户病例严格隔离。"""

    def test_user_case_isolation(self, tmp_path, monkeypatch):
        """不同用户看不到彼此的病例。"""
        from medrag.data.user_case_store import _case_store as original_store
        store_path = tmp_path / "user_cases.json"
        from medrag.infrastructure.storage import JsonStore
        mock_store = JsonStore(str(store_path))
        monkeypatch.setattr("medrag.data.user_case_store._case_store", mock_store)

        add_user_case("alice", "a.pdf", ["Alice专属报告：甘油三酯 3.2 mmol/L"], summary="Alice病例")
        add_user_case("bob", "b.pdf", ["Bob专属报告：肌酸激酶 180 U/L"], summary="Bob病例")

        alice_cases = get_user_cases("alice")
        bob_cases = get_user_cases("bob")

        assert len(alice_cases) == 1
        assert len(bob_cases) == 1
        assert alice_cases[0]["filename"] == "a.pdf"
        assert bob_cases[0]["filename"] == "b.pdf"

        # 验证个人检索结果不交叉
        retriever = UserCaseRetriever()
        alice_results = retriever.search("甘油三酯", username="alice")
        bob_results = retriever.search("甘油三酯", username="bob")
        assert len(alice_results) > 0
        assert len(bob_results) == 0

    def test_add_user_case_replaces_existing(self, tmp_path, monkeypatch):
        """同一用户上传同名文件应替换旧记录。"""
        from medrag.infrastructure.storage import JsonStore
        store_path = tmp_path / "user_cases_replace.json"
        monkeypatch.setattr("medrag.data.user_case_store._case_store",
                            JsonStore(str(store_path)))

        add_user_case("alice", "same.pdf", ["v1 data"], summary="v1")
        add_user_case("alice", "same.pdf", ["v2 data"], summary="v2")

        cases = get_user_cases("alice")
        assert len(cases) == 1  # 替换，不新增
        assert cases[0]["summary"] == "v2"


# ============================================================================
# 正常查询不强行引用病例（MedicalChatService 级别）
# ============================================================================


class TestNoForcedCaseReference:
    """验证没有病例上下文时，chat_service 不会强行引用病例。"""

    def test_chat_without_username_no_case(self):
        """不传 username 时不应查询病例。"""
        from medrag.rag.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        prompt = builder.build_answer_prompt(
            query="感冒了怎么办",
            route={"query_type": "general_medical_qa", "answer_style": "general_guidance"},
        )

        assert "（此部分暂无可用资料）" in prompt
        # 没有 case_context 时，病例摘要为空
        assert "## 用户病例摘要" in prompt  # section 存在
        assert "## 用户病例片段" in prompt  # section 存在

    def test_chat_without_case_context_no_case_text(self):
        """case_context 为 None 时，病例摘要应为空占位。"""
        from medrag.rag.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        prompt = builder.build_answer_prompt(
            query="高血压注意事项",
            case_context=None,
            route={"query_type": "general_medical_qa", "answer_style": "general_guidance"},
        )

        # 应显示空占位，不应有实际病例内容
        assert "（此部分暂无可用资料）" in prompt


class TestPublicQASourceWhitelist:
    def test_public_qa_filters_source_and_escapes_department(self):
        from medrag.vectors.qa_retriever import QARetriever

        retriever = QARetriever.__new__(QARetriever)
        retriever._available = True
        retriever.embedding_model = MagicMock()
        retriever.embedding_model.encode_one.return_value = [0.1, 0.2]
        retriever._milvus = MagicMock()
        retriever._milvus.search.return_value = [
            {
                "id": "public",
                "distance": 0.9,
                "entity": {
                    "pk": "public",
                    "source": "cmedqa2",
                    "answer": "public answer",
                },
            },
            {
                "id": "private",
                "distance": 0.99,
                "entity": {
                    "pk": "private",
                    "source": "private-report.pdf",
                    "answer": "private case",
                },
            },
        ]

        department = 'cardiology" or source != "cmedqa2'
        results = retriever.search("query", top_k=5, department=department)

        expr = retriever._milvus.search.call_args.kwargs["expr"]
        assert 'source == "cmedqa2"' in expr
        assert "source" in retriever._milvus.search.call_args.kwargs["output_fields"]
        assert 'department == "cardiology\\" or source != \\"cmedqa2"' in expr
        assert [result["id"] for result in results] == ["public"]


class TestLegacyDocumentIsolation:
    def test_document_without_username_is_hidden_and_preserved(
        self, tmp_path, monkeypatch
    ):
        from medrag.app import document_store

        store_path = tmp_path / "legacy_docs_index.json"
        store = document_store.JsonStore(str(store_path))
        store.write(
            [
                {
                    "filename": "legacy.pdf",
                    "file_type": "PDF",
                    "chunk_count": 1,
                },
                {
                    "filename": "owned.pdf",
                    "file_type": "PDF",
                    "chunk_count": 1,
                    "username": "alice",
                },
            ]
        )
        monkeypatch.setattr(document_store, "_doc_store", store)

        visible = document_store.get_documents(username="alice")

        assert [document.filename for document in visible] == ["owned.pdf"]
        assert (
            document_store.get_document_by_filename("legacy.pdf", username="alice")
            is None
        )
        assert document_store.remove_document("legacy.pdf", username="alice") is False
        assert [doc["filename"] for doc in document_store._read_docs()] == [
            "legacy.pdf",
            "owned.pdf",
        ]
