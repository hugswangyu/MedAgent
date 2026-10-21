"""Small dependency-aware executor for the medical Harness workflow."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional


Handler = Callable[[Dict[str, Any]], Any]
Condition = Callable[[Dict[str, Any], Dict[str, Any]], bool]


@dataclass
class DAGNode:
    name: str
    handler: Handler
    dependencies: set[str] = field(default_factory=set)
    condition: Optional[Condition] = None
    timeout_ms: Optional[int] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    pause_on_success: bool = False


@dataclass
class DAGNodeResult:
    status: str = "success"
    data: Any = None
    error: Optional[str] = None
    dynamic_nodes: List[DAGNode] = field(default_factory=list)


@dataclass
class DAGExecution:
    status: str
    outputs: Dict[str, Any]
    trace: List[Dict[str, Any]]
    waiting_for_user: bool = False
    question: Optional[str] = None


class DAGExecutor:
    """Execute dependency-ready nodes in parallel and support runtime insertion."""

    def __init__(
        self,
        nodes: Iterable[DAGNode] = (),
        *,
        max_workers: int = 4,
        default_timeout_ms: Optional[int] = None,
    ) -> None:
        self._nodes: Dict[str, DAGNode] = {}
        self._lock = RLock()
        self.max_workers = max(1, max_workers)
        self.default_timeout_ms = default_timeout_ms
        for node in nodes:
            self.add_node(node)

    @property
    def nodes(self) -> Dict[str, DAGNode]:
        return dict(self._nodes)

    def add_node(self, node: DAGNode) -> None:
        with self._lock:
            if node.name in self._nodes:
                raise ValueError(f"DAG node already exists: {node.name}")
            self._nodes[node.name] = node
            self._validate()

    def insert_after(self, after: str, node: DAGNode) -> None:
        """Insert a node at runtime while preserving an acyclic dependency graph."""
        if after not in self._nodes:
            raise KeyError(f"Unknown DAG dependency: {after}")
        node.dependencies = set(node.dependencies) | {after}
        self.add_node(node)

    def run(self, context: Optional[Dict[str, Any]] = None) -> DAGExecution:
        context = context if context is not None else {}
        outputs: Dict[str, Any] = {}
        trace: List[Dict[str, Any]] = []
        completed: set[str] = set()
        skipped: set[str] = set()
        failed = False

        self._validate()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while len(completed | skipped) < len(self._nodes):
                for name, node in self._nodes.items():
                    if (
                        name not in completed
                        and name not in skipped
                        and node.dependencies <= completed
                        and node.condition is not None
                        and not node.condition(context, outputs)
                    ):
                        skipped.add(name)
                        trace.append({"node": name, "status": "skipped", "error": None})

                ready = self._ready_nodes(completed | skipped, context, outputs)
                if not ready:
                    pending = set(self._nodes) - completed - skipped
                    raise ValueError(f"DAG has unresolved dependencies or a cycle: {sorted(pending)}")

                futures = {
                    node.name: pool.submit(self._execute_node, node, context)
                    for node in ready
                }
                pause_nodes: List[str] = []

                for name, future in futures.items():
                    node = self._nodes[name]
                    result = future.result()
                    completed.add(name)
                    context.setdefault("dag_outputs", {})[name] = result.data
                    outputs[name] = result.data
                    trace.append({
                        "node": name,
                        "status": result.status,
                        "error": result.error,
                    })
                    failed = failed or result.status == "failed"

                    for dynamic_node in result.dynamic_nodes:
                        self.insert_after(name, dynamic_node)

                    if node.pause_on_success and result.status == "success":
                        pause_nodes.append(name)

                if pause_nodes:
                    question = context.get("clarifying_question")
                    return DAGExecution(
                        status="paused",
                        outputs=outputs,
                        trace=trace,
                        waiting_for_user=True,
                        question=question,
                    )

        return DAGExecution(
            status="failed" if failed else "success",
            outputs=outputs,
            trace=trace,
        )

    def _ready_nodes(
        self,
        completed: set[str],
        context: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[DAGNode]:
        ready = []
        for node in self._nodes.values():
            if node.name in completed:
                continue
            if node.dependencies <= completed and (
                node.condition is None or node.condition(context, outputs)
            ):
                ready.append(node)
        return ready

    def _execute_node(self, node: DAGNode, context: Dict[str, Any]) -> DAGNodeResult:
        policy = {"max_retries": 0, "delay_ms": 0, "backoff": 1.0}
        policy.update(node.retry_policy)
        attempts = int(policy["max_retries"]) + 1
        timeout_ms = node.timeout_ms or self.default_timeout_ms
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(node.handler, context)
                value = future.result(timeout=timeout_ms / 1000 if timeout_ms else None)
                if isinstance(value, DAGNodeResult):
                    return value
                return DAGNodeResult(data=value)
            except TimeoutError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

            if attempt < attempts - 1:
                import time
                delay = float(policy["delay_ms"]) / 1000
                delay *= float(policy["backoff"]) ** attempt
                time.sleep(max(delay, 0))

        return DAGNodeResult(
            status="failed",
            error=f"{type(last_error).__name__}: {last_error}",
        )

    def _validate(self) -> None:
        names = set(self._nodes)
        for node in self._nodes.values():
            unknown = node.dependencies - names
            if unknown:
                raise ValueError(f"DAG node {node.name} has unknown dependencies: {sorted(unknown)}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"DAG cycle detected at node: {name}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self._nodes[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in names:
            visit(name)
