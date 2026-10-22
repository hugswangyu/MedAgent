"""回答自然度和真实检索状态的回归测试，不访问数据库或云端模型。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from medrag.harness.orchestrator import HarnessOrchestrator
from medrag.rag.safety_guard import SafetyGuard
from medrag.react.rag_tool import RetrieveKnowledgeTool
from medrag.retrieval.hybrid_retriever import HybridRetriever
from medrag.retrieval.router import QueryRouter


@pytest.mark.parametrize('query', ['hello', 'Hello!', '你好！', '谢谢你', '再见', '你能帮我做什么？'])
def test_pure_smalltalk_skips_llm_routing_and_retrieval(query, monkeypatch):
    router = QueryRouter()
    remote = Mock(side_effect=AssertionError('纯问候无需远程路由'))
    monkeypatch.setattr(router, '_llm_route', remote)
    route = router.route(query)
    assert route['query_type'] == 'smalltalk'
    assert not route['use_kg'] and not route['use_qa']
    assert not route['needs_case_context']
    remote.assert_not_called()


@pytest.mark.parametrize('query', ['你好，我胸痛怎么办？', '谢谢，但阿莫西林还能吃吗？'])
def test_greeting_with_medical_request_is_not_smalltalk(query):
    assert QueryRouter().route(query, use_llm=False)['query_type'] != 'smalltalk'


def test_llm_smalltalk_route_does_not_force_a_retrieval_source():
    route = QueryRouter._parse_llm_response('{"query_type":"smalltalk","kg":false,"qa":false}')
    assert not route['use_qa'] and not route['use_kg']


def run_answer(query, answer, retrieval=None, failure=False, unrelated_tool=False):
    trace = {}
    retriever = Mock()
    retriever.retrieve.return_value = retrieval
    if failure:
        retriever.retrieve.side_effect = RuntimeError('offline')
    tool = RetrieveKnowledgeTool(retriever, Mock(), trace_context=trace)
    guard = SafetyGuard()

    def run(*args, **kwargs):
        steps, results = [], {}
        if retrieval is not None or failure:
            observation = tool.execute(query)
            steps = [{'action': 'retrieve_knowledge'}]
            results = {'retrieve_knowledge': [observation]}
        elif unrelated_tool:
            steps = [{'action': 'normal_range'}]
            results = {'normal_range': ['正常值范围']}
        return {'answer': answer, 'steps': steps, 'tool_results': results}

    def safety(query, answer, retrieval_quality, query_type):
        return guard.append_safety_notice(answer, guard.detect_risk(query), retrieval_quality, query_type)

    result = HarnessOrchestrator(
        router=lambda q: QueryRouter().route(q, use_llm=False),
        risk_detector=guard.detect_risk,
        react_engine_builder=lambda *args: SimpleNamespace(run=run),
        safety_checker=safety,
        retrieval_quality_getter=lambda: trace.get('retrieval_quality', 'not_requested'),
    ).run(query)
    return result


def test_hello_answer_is_not_modified_by_medical_footer():
    answer = '你好！有什么可以帮你？'
    assert run_answer('hello', answer)['answer'] == answer


@pytest.mark.parametrize('unrelated_tool', [False, True])
def test_no_knowledge_retrieval_does_not_claim_empty_or_high_quality(unrelated_tool):
    answer = '请补充问题的具体情况。'
    result = run_answer('头痛怎么办', answer, unrelated_tool=unrelated_tool)
    assert result['answer'] == answer
    assert result['retrieval_quality'] == 'not_requested'


def test_empty_retrieval_explains_missing_evidence_once():
    result = run_answer('头痛怎么办', '暂时无法判断。', retrieval={})
    assert result['retrieval_quality'] == 'none'
    assert '未检索到能直接回答' in result['answer']
    assert '请咨询专业医生' not in result['answer']
    answer = '目前资料不足，请补充头痛的持续时间。'
    assert run_answer('头痛怎么办', answer, retrieval={})['answer'] == answer


def test_failed_retrieval_is_not_reported_as_no_match():
    result = run_answer('头痛怎么办', '暂时无法判断。', failure=True)
    assert result['retrieval_quality'] == 'error'
    assert '暂时无法查询' in result['answer']
    assert '未检索到' not in result['answer']


def test_real_retrieval_does_not_get_a_fixed_footer():
    answer = '参考资料中列出了常见诱因。'
    result = run_answer('头痛怎么办', answer, retrieval={'kg_results': [{'answer': '常见诱因'}]})
    assert result['retrieval_quality'] == 'retrieved'
    assert result['answer'] == answer


@pytest.mark.parametrize('query,warning', [('你好，我胸痛怎么办？', '120'), ('我有便血怎么办', '尽快就医')])
def test_risk_warnings_remain_even_without_retrieval(query, warning):
    assert warning in run_answer(query, '请及时就医。')['answer']


def test_backend_failures_are_preserved_as_structured_status():
    broken = Mock()
    broken.search.side_effect = RuntimeError('offline')
    hybrid = HybridRetriever(kg_retriever=broken, qa_retriever=broken, es_retriever=broken)
    retrieval = hybrid.retrieve('头痛', route={'use_kg': True, 'use_qa': True})
    assert set(retrieval['retrieval_errors']) == {'neo4j', 'milvus', 'es'}
    result = run_answer('头痛怎么办', '暂时无法判断。', retrieval=retrieval)
    assert result['retrieval_quality'] == 'error'


def test_partial_backend_failure_is_incomplete_instead_of_high_quality():
    result = run_answer('头痛怎么办', '找到一条相关资料。', retrieval={
        'kg_results': [{'answer': '相关资料'}], 'retrieval_errors': ['milvus'],
    })
    assert result['retrieval_quality'] == 'low'
    assert '参考资料不完整' in result['answer']


@pytest.mark.parametrize('retrieval,quality', [({}, 'none'), ({'kg_results': [{'answer': '资料'}]}, 'retrieved')])
def test_chat_service_passes_request_local_retrieval_status(retrieval, quality):
    from medrag.service.chat_service import MedicalChatService

    service = MedicalChatService.__new__(MedicalChatService)
    service.hybrid_retriever = SimpleNamespace(router=QueryRouter())
    service.hybrid_retriever.router.route = lambda query: {'query_type': 'disease_fact'}
    service.safety_guard = SafetyGuard()
    service._get_request_memory = Mock(return_value=Mock(build_context=Mock(return_value='')))
    service._get_tool_registry = Mock(return_value=Mock(match=Mock(return_value=(None, None))))
    service._get_query_embedding = Mock(return_value=None)
    service._compose_system_context = Mock(return_value='')

    def builder(query, route, **kwargs):
        tool = RetrieveKnowledgeTool(Mock(retrieve=Mock(return_value=retrieval)), Mock(),
                                     trace_context=kwargs['trace_context'])

        def run(*args, **kwargs):
            observation = tool.execute(query)
            return {'answer': '这是回答。', 'steps': [{'action': 'retrieve_knowledge'}],
                    'tool_results': {'retrieve_knowledge': [observation]}}
        return SimpleNamespace(run=run)

    service._build_react_engine = builder
    result = service.chat_with_harness('头痛是什么原因')
    assert ('未检索到' in result['answer']) == (quality == 'none')
