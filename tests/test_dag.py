"""Tests for the dependency-aware Harness DAG executor."""
from __future__ import annotations

import time

from medrag.harness.dag import DAGExecutor, DAGNode, DAGNodeResult


def test_dag_runs_dependency_ready_nodes_and_records_trace():
    events = []

    def first(context):
        events.append("first")
        return "one"

    def second(context):
        events.append("second")
        return "two"

    def join(context):
        events.append("join")
        return "joined"

    execution = DAGExecutor([
        DAGNode("first", first),
        DAGNode("second", second),
        DAGNode("join", join, {"first", "second"}),
    ]).run()

    assert execution.status == "success"
    assert execution.outputs["join"] == "joined"
    assert events[-1] == "join"
    assert {item["node"] for item in execution.trace} == {"first", "second", "join"}


def test_dag_rejects_cycles():
    try:
        DAGExecutor([
            DAGNode("a", lambda context: None, {"b"}),
            DAGNode("b", lambda context: None, {"a"}),
        ])
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("expected a cycle validation error")


def test_dag_can_insert_a_pause_node_after_runtime_result():
    def react(context):
        context["clarifying_question"] = "请补充症状持续时间。"
        return DAGNodeResult(dynamic_nodes=[
            DAGNode(
                "clarify_question",
                lambda inner: inner["clarifying_question"],
                pause_on_success=True,
            ),
        ])

    execution = DAGExecutor([
        DAGNode("react", react),
    ]).run()

    assert execution.status == "paused"
    assert execution.waiting_for_user is True
    assert execution.question == "请补充症状持续时间。"
    assert execution.outputs["clarify_question"] == "请补充症状持续时间。"


def test_dag_node_timeout_returns_a_failed_result():
    def slow(context):
        time.sleep(0.05)
        return "late"

    execution = DAGExecutor([
        DAGNode("slow", slow, timeout_ms=1),
    ]).run()

    assert execution.status == "failed"
    assert execution.trace[0]["status"] == "failed"
