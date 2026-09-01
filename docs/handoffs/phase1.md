# MedAgent × LiveRAG 阶段 1 交接记录

日期：2026-09-01  
阶段：身份与数据基础  
状态：代码、单元测试、静态检查与 PostgreSQL migration 验收完成；真实语音验收仍被外部凭据阻断  
提交：未提交

## 实施边界

1. 两个仓库继续位于 D:MedAgent 与 D:LiveRAG，没有迁仓。
2. 保留 MedAgent 用户原有未提交修改：
   - src/medrag/config/settings.py 的 load_dotenv(..., override=False)；
   - fastapi-startup.err、fastapi-startup.log、handoff.md。
3. 旧 user_credentials.json 只作为一次性导入源读取；不会被阶段 1 删除、覆盖或改写。
4. 阶段 0 的四个 /internal/v1 能力路径、超时、幂等键和 envelope 保持兼容。
5. 生产环境默认不再接受共享 internal API key；开发/迁移期仍可显式启用。

## 已完成

### PostgreSQL 用户与 user_id

- 新增 users 表，以 UUID user_id 为主键，username 只作为唯一登录名。
- MedAgent 启动时幂等安装阶段 1 schema，并把旧 JSON 用户复制到 PostgreSQL。
- 旧明文密码只在内存中转为 bcrypt 后写入 PostgreSQL，旧文件保持原样。
- 登录、注册、/auth/me 响应均返回 user_id。
- 访问 JWT 的 sub 已改为 user_id，另带只读 username claim；依赖按 user_id 查 PostgreSQL 用户。

### 知识库所有权

- LiveRAG knowledge_bases 增加 owner_user_id，已有 SQLite 数据采用幂等 ALTER TABLE 升级。
- 管理 API 的知识库列表、详情、修改、删除、文档、任务、预览和查询路径均按 JWT 当前 user_id 检查所有权。
- 跨用户读取与“不存在”统一返回 404，避免通过错误差异枚举他人知识库。
- 会话默认知识库配置改为 knowledge_base:{user_id}，不再由不同用户共享同一个配置键。
- 升级前旧知识库不会被自动暴露给任意新用户。管理员需把既有 PostgreSQL users.user_id 配到 LIVERAG_LEGACY_OWNER_USER_ID 完成一次性认领；不配置时文件和索引原样保留但保持未认领。

### Voice Session 服务端绑定

- LiveRAG voice_sessions 增加 user_id，创建、读取、刷新 token、turn、RAG context 和结束接口均校验当前用户。
- 每个用户独立判断活动会话，不再由其他用户的活动通话阻塞。
- room/job metadata 只允许提供 session_id。worker 必须回查服务端 SQLite 记录，并校验实际 room、会话状态、user_id 和 kb_id；metadata 中伪造的用户或知识库字段不会生效。
- 运行状态写入当前 user_id，按用户读取活动知识库锁定。

### turn、evidence 与审计

scripts/phase1_identity_data.sql 新增：

- voice_sessions
- voice_turns
- evidence
- audit_events
- worker_request_nonces

worker 调用 MedAgent 能力时：

- 先把 session_id 原子绑定到 token 的 sub 与 kid；
- 每轮按 (session_id, turn_id) 写入或确认 voice_turns 绑定，允许不同会话各自使用 turn_1；
- 医学检索及输出检查携带的统一 Evidence 写入 evidence；
- 操作、结果、request id、延迟与错误码写入 audit_events；
- 审计详情主动排除 text、query、content，不记录原始病例或完整模型文本。

### 短期 worker token 与防重放

- LiveRAG 与 MedAgent 使用同一个显式 JWT_SECRET_KEY。
- worker token 使用 HS256，aud=medagent-internal，token_use=worker，默认有效期 300 秒。
- token 必含 sub=user_id、sid=voice_session_id、kid=knowledge_base_id、scope、jti、iat、exp。
- LiveRAG worker 按请求签发短期 token，并为每个 HTTP 请求生成新的 UUID X-Request-Nonce。
- MedAgent 校验签名、audience、scope、sid 和 UUID 格式后，通过 PostgreSQL (token_jti, nonce) 主键原子消费 nonce。
- 同一 token 与 nonce 的重放返回 REPLAY_DETECTED；使用新 nonce 的同幂等请求仍由阶段 0 幂等层返回首次结果。
- 生产环境默认关闭共享 key；短期迁移可显式设置 MEDAGENT_ALLOW_LEGACY_INTERNAL_API_KEY=true。

## 关键文件

MedAgent：

- scripts/phase1_identity_data.sql
- src/medrag/infrastructure/storage/phase1_repository.py
- src/medrag/app/auth_manager.py
- src/medrag/app/worker_auth.py
- src/medrag/app/api/internal_v1.py
- tests/test_phase1_identity_data.py

LiveRAG：

- liverag/security.py
- liverag/rag/metadata_store.py
- liverag/voice/session.py
- liverag/api/server.py
- liverag/main.py
- liverag/agent/tool/medical_client.py
- tests/test_phase1_identity.py

## 部署与升级顺序

1. 备份 PostgreSQL 与 ~/.LiveRAG/，不要删除旧凭据、SQLite、知识库目录或索引。
2. 在 MedAgent PostgreSQL 执行 scripts/phase1_identity_data.sql，并启用 ON_ERROR_STOP。
3. 两端配置同一个随机强 JWT_SECRET_KEY。
4. 启动 MedAgent。它会再次幂等检查 schema，并把旧 JSON 用户导入 users。
5. 查询要承接旧 LiveRAG 知识库的 users.user_id，配置 LIVERAG_LEGACY_OWNER_USER_ID。
6. 首次启动 LiveRAG 管理 API/RAG 服务，完成 SQLite owner_user_id、Voice Session user_id 列升级与旧库认领。
7. 确认登录返回的 JWT sub 是 UUID，再验证用户只能看到和操作自己的知识库。
8. 生产环境保持 MEDAGENT_ALLOW_LEGACY_INTERNAL_API_KEY=false 或不配置；确认 worker 请求使用 Bearer token 与 X-Request-Nonce。

## 验证结果

MedAgent 阶段 0/1 定向回归：

    31 passed, 1 warning

MedAgent 全量：

    218 passed, 11 failed, 1 warning

11 个失败与阶段 0 交接一致，仍位于本阶段未修改的旧模块：

- tests/test_eval_script.py：7 个；
- tests/test_memory_system.py：2 个；
- tests/test_query_routing_cases.py：1 个；
- tests/test_resilience.py：1 个。

LiveRAG：

    uv run pytest -q
    22 passed

    uv run ruff check liverag tests
    All checks passed!

    uv run python -m compileall -q liverag tests
    通过

两仓 git diff --check 与 Python compileall 均通过。

PostgreSQL 16 临时真实实例：

- 首次 migration 创建 users、voice_sessions、voice_turns、evidence、audit_events、worker_request_nonces；
- 第二次 migration 成功，验证幂等；
- 表清单核对通过；
- 两个不同 session 均成功写入 turn_1，验证 (session_id, turn_id) 复合主键不会跨会话冲突；
- 验收临时容器已停止并由 --rm 自动删除。

LiveRAG OpenAPI smoke test 确认 /voice/sessions 与 /rag/knowledge-bases 均带 Bearer security requirement；临时运行目录已清理。

## 明确未完成：真实语音验收

**真实语音验收没有通过，也没有被本阶段声明为通过。**

当前仍缺少或未提供 LiveKit、真实 STT、Voice LLM、TTS 等外部凭据，因此只完成了代码、HTTP 契约、token、nonce、SQLite/PostgreSQL 与离线测试。

凭据提供后必须继续执行：

1. 同时启动 PostgreSQL、MedAgent、LiveRAG 管理 API/RAG service、LiveKit worker 和真实语音 provider。
2. 用两个真实登录用户验证知识库列表、文档、查询和 Voice Session 全链路互相不可见。
3. 抓取真实 worker 请求，确认 sub/sid/kid/aud/scope/jti、五分钟有效期和每请求 nonce。
4. 重放同一 Bearer token + nonce，确认被 PostgreSQL 防重放拒绝；以新 nonce 重试同一幂等请求，确认返回首次结果。
5. 用实际音频复验阶段 0 急症旁路、安全 TTS、医学检索、个人库和三个确定性工具。
6. 采集首响、100 turn、15 分钟并发/稳定性与审计表对账数据。

在上述工作完成前，不得把“真实麦克风音频”“真实 provider”或“生产端到端”标记为已验收。
