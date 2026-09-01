# MedAgent × LiveRAG 阶段 0 交接记录

日期：2026-09-01  
状态：代码、HTTP 与本地契约验收完成；真实音频验收待凭据  
提交：未提交

## 完成范围

1. 两个仓库保持在 `D:\MedAgent` 与 `D:\LiveRAG`，没有迁移源码。
2. MedAgent 新增四个 `/internal/v1` POST 能力接口：
   - `/safety/input-check`
   - `/safety/output-check`
   - `/medical/retrieve`
   - `/medical/tools/execute`
3. MedAgent 医学检索为 retrieval-only，强制排除用户病例检索，不调用 LLM/答案生成。
4. 输入安全区分当前本人、他人、历史、否定和科普；当前红色急症返回固定响应。
5. 输出安全返回唯一可播报 `safe_text`；结构化错误、硬超时和幂等语义已实现。
6. LiveRAG 唯一 `VoiceAssistant` 通过 HTTP 接入医学能力，固定暴露五个工具名。
7. 个人库和医学库证据统一为 Evidence 字段集合，不设置来源硬编码优先级。
8. LiveRAG 在默认 LLM node 前执行输入检查，在真实 TTS node 前执行句级/120 字安全缓冲。
9. 输入或输出安全服务异常时 fail-closed，原始模型文本不会进入 TTS。
10. 个人文档明确作为不可信数据；所有会话（含使用旧持久化模板者）强制追加阶段 0 安全规则。
11. 未实施用户迁移、PostgreSQL、统一记忆、迁仓、前端替换或后续阶段内容。
12. 并发相同幂等请求采用 single-flight，只执行一次后端操作。
13. 两端均移除默认内部密钥；MedAgent 未显式配置时拒绝请求，LiveRAG 拒绝空 key。
14. LiveRAG 复用单个 HTTP session 并在语音会话结束时关闭。
15. 模型文本在 `llm_node` 内检查后才进入任何下游，TTS 前再检查一次；日志不记录原始模型预览。
16. 医学检索强制规则路由 `use_llm=false`，杜绝路由阶段调用 LLM。
17. 输入安全按红色症状语义分片判断；混合语境中的当前本人急症优先于历史/他人描述。

## MedAgent 文件

新增：

- `src/medrag/contracts/__init__.py`
- `src/medrag/contracts/phase0.py`
- `src/medrag/service/phase0_capabilities.py`
- `src/medrag/app/api/internal_v1.py`
- `tests/test_phase0_contracts.py`
- `docs/phase0-contract.md`
- `docs/handoffs/phase0.md`

修改：

- `src/medrag/app/server.py`：注册 internal v1、注入现有 MedicalChatService、统一 internal 参数验证错误。
- `.env.example`：内部 API key 与四类硬超时。

保护：

- 未改动用户原有 `src/medrag/config/settings.py` 未提交修改。
- 未改动原有 `fastapi-startup.err`、`fastapi-startup.log`、`handoff.md`。
- 未提交或清理任何用户内容。

## LiveRAG 文件

新增：

- `liverag/agent/tool/medical_client.py`
- `liverag/agent/safety.py`
- `liverag/agent/stepfun_stt.py`
- `liverag/agent/stepfun_tts.py`
- `tests/test_phase0_medical_integration.py`
- `tests/test_stepfun_voice.py`
- `docs/phase0-contract.md`

修改：

- `liverag/agent/assistant.py`
- `liverag/agent/tool/__init__.py`
- `liverag/agent/tool/rag_client.py`
- `liverag/context/defaults.py`
- `liverag/context/renderer.py`
- `liverag/main.py`
- `.env.example`

没有修改 `AgentSession` 的 STT、LLM、TTS、VAD、打断、preemptive generation 或 endpointing 调优参数。

## 冻结语义

详见：

- MedAgent：`D:\MedAgent\docs\phase0-contract.md`
- LiveRAG：`D:\LiveRAG\docs\phase0-contract.md`

关键口径：

- 输入安全 400ms。
- 医学检索 1500ms。
- 个人库检索 2000ms（LiveRAG 现有独立配置/降级）。
- 确定性医疗工具 500ms。
- 输出安全 400ms。
- 相同操作 + 相同幂等键 + 相同 payload 重放首次结果；不同 payload 返回 `IDEMPOTENCY_CONFLICT`。
- 鉴权失败为 HTTP 401 envelope；模型验证失败为 HTTP 422 `INVALID_REQUEST` envelope；能力错误为 HTTP 200 error envelope。

## 验证结果

MedAgent：

```text
python -m compileall src\medrag
通过

pytest -q tests\test_phase0_contracts.py
8 passed, 1 warning
```

警告为 FastAPI/Starlette TestClient 的可选 `httpx2` 迁移提示，不影响运行契约。

LiveRAG：

```text
uv run ruff check liverag tests
All checks passed

uv run python -m compileall liverag tests
通过

uv run pytest -q
17 passed
```

两仓 `git diff --check` 均通过。MedAgent OpenAPI smoke test确认四个 internal v1 路径存在。LiveRAG 测试包含真实本地 HTTP server/client 往返，验证 API key、通用关联字段、幂等键和 envelope 解析。

LiveRAG 另已按 Step Plan 官方协议加入可选 `stepaudio-2.5-asr` HTTP+SSE 与 `stepaudio-2.5-tts` WebSocket PCM 适配器；协议请求、SSE 解析、provider 构建和环境映射离线测试通过。该路径在用户配置真实 `STEPFUN_API_KEY` 前仍标记为未完成真实音频验收。

MedAgent 完整旧测试：

```text
211 passed, 11 failed
```

11 个失败位于本轮未修改的旧模块：

- `tests/test_eval_script.py`：7 个，评测辅助函数返回/指标契约不一致。
- `tests/test_memory_system.py`：2 个，偏好识别旧测试失败。
- `tests/test_query_routing_cases.py`：1 个，Milvus wrapper 旧 mock/客户端属性不一致。
- `tests/test_resilience.py`：1 个，QARetriever 旧失败态属性不一致。

这些失败不经过阶段 0 新接口，未越界修复。

## 真实进程验收结果

使用随机临时内部密钥启动真实 `uvicorn` MedAgent 进程，并通过本机 TCP/HTTP 完成：

- 缺失 key 返回 HTTP 401。
- “胸痛是什么”返回 `allow/educational`。
- “我现在胸痛而且呼吸困难”返回 `emergency`，固定响应包含 120。
- 危险输出“不用去急诊，在家等一等。”被替换；安全输出允许通过。
- 导诊、剂量和正常值三种确定性医疗工具均返回固定可审计结果。
- 相同幂等请求返回同一 `request_id`，重放标志正确。
- Docker 启动 Elasticsearch/Milvus 后，医学检索返回 `retrieval_only=true`、6 条证据，服务端 `latency_ms=304.6`，低于 1500ms 硬超时。

首次检索曾因 Elasticsearch/Milvus 未启动而在约 1503ms 超时；依赖启动且规则路由修复加载后已通过。运行日志未再触发 DeepSeek 路由调用。

验收完成后已停止临时 MedAgent 进程和本次启动的 Docker 容器，并删除临时随机密钥。日志保留在 `C:\Users\MSPZ\AppData\Local\Temp\medagent-phase0-acceptance` 供追溯。

## 真实语音外部阻断

当前 `D:\LiveRAG` 没有 `.env`，且以下运行配置均缺失：LiveKit URL/API key/API secret、火山 STT app ID/token、Voice LLM API key、TTS API key。没有 LiveKit、语音 worker、STT/LLM/TTS provider 端口在运行。因此不能诚实地把本地 HTTP/契约测试记为“真实麦克风音频已通过”。

提供上述配置后仍需执行：

1. 同时启动 MedAgent API、LiveRAG RAG service、LiveKit worker 和真实语音 provider。
2. 用实际音频验证急症输入旁路 LLM/RAG，以及安全固定文本首段播报。
3. 用实际音频调用医学检索、个人库和三种确定性医疗工具。
4. 注入个人文档提示词，并抓取真实 TTS 输入确认无未检查模型片段。
5. 按计划采集首响、100 turn 和 15 分钟并发/稳定性数据。

## 已知阶段 0 限制

- 内部 API key 是共享密钥；短期 worker JWT、scope、audience 和 Voice Session 服务端绑定属于阶段 1。
- 幂等缓存为单进程内存，服务重启后清空；持久化幂等与审计属于后续数据阶段。
- turn、证据、安全结果尚未写入 PostgreSQL；当前仅沿用 LiveRAG 运行记录。
- 规则式安全检查用于验证链路和契约，不等同于生产级临床安全分类器。
- internal API key 无默认值，两端必须显式配置同一随机密钥。

## 下一步门槛

只有完成“真实语音外部阻断”中的端到端音频验收，并确认 OpenAPI、错误码、超时和幂等语义无需变更后，才进入阶段 1。不得从本交接直接开始迁仓、用户迁移、统一记忆或前端替换。
