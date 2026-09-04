# MedAgent × LiveRAG Phase 4 交接记录

日期：2026-09-02
阶段：迁仓与统一前端
状态：代码迁移、离线回归和 Next.js 生产构建通过；真实 PostgreSQL、LiveKit 与语音 provider 端到端仍待凭据
提交：`e0b45ba`（Phase 4 迁仓与统一前端）

## Phase 3 前置状态

Phase 3 原先没有独立交接文件。Phase 4 开始前运行定向测试得到 24 passed、1 skipped；跳过项为真实 PostgreSQL 集成。详情见 docs/handoffs/phase3.md。

## 完成范围

1. 将 LiveRAG 稳定 Python 模块复制到 src/medlive，并把内部包导入从 liverag 改为 medlive。
2. 新增 src/medcontracts，作为冻结请求、证据、错误和事件 envelope 的共享契约实现。
3. medrag.contracts 保留兼容 shim；MedAgent internal API 和能力服务直接导入 medcontracts。
4. MedLive 医学客户端使用 CapabilityEnvelope 校验 MedAgent 响应，不再维护重复 envelope 解析。
5. pyproject.toml 注册 medlive-agent、medlive-api、medlive-rag-service，并把 LiveKit/LightRAG 运行依赖放入可选 live 依赖组；本阶段未部署这些服务。
6. 以 LiveRAG Next.js 前端替换根 frontend，接入统一登录、文字问诊、实时语音、个人知识库、证据和受控记忆管理。
7. Next.js 通过 HttpOnly、SameSite=Lax cookie 保存 MedAgent access token；浏览器只访问同源代理。
8. /api/medagent 和 /api/liverag 代理从服务端 cookie 注入 Bearer，不把 JWT 暴露给浏览器脚本。
9. /api/token 不再自行签发不受绑定的 LiveKit token；它先读取当前个人库，再调用 MedLive /voice/sessions 创建服务端绑定会话并映射为 Agents UI 连接信息。
10. 原 Vue 文件原样移动到 frontend-legacy，并复制到 Next.js public/legacy；/legacy 重定向到 /legacy/index.html。根路径只挂载统一 Next.js。
11. 保留 LiveRAG MIT 全文于 licenses/LiveRAG-MIT.txt，并新增 THIRD_PARTY_NOTICES.md。
12. 没有导入 D:\LiveRAG 的 Git 历史，没有修改其仓库。

## LiveRAG 来源指纹

- 源 commit：a5124c95c8b11a2208177cc075648e01826a92bc
- 复制时源仓 ahead 5
- 复制时包含三项源工作区修改：
  - liverag/context/history.py
  - liverag/context/renderer.py
  - liverag/voice/session.py
- 这些文件属于 Phase 3 停用独立 history 和统一会话语义所需的稳定基线，按当时工作树内容复制。
- 源码通过文件复制进入主仓，不包含 .git 或提交历史。

## 统一前端入口

根页面 frontend/app/page.tsx 挂载 UnifiedApp。登录后提供五个入口：

- 文字问诊：调用 MedAgent /chat/stream，并保留 SSE 中的证据或检索轨迹。
- 实时语音：复用 LiveRAG Agents UI，由绑定式 MedLive voice session 提供连接信息。
- 个人知识库：复用 LiveRAG 知识库和文档管理界面，所有请求携带同一用户身份。
- 证据：读取最近语音 turns 和 runtime state，展示当前知识库证据。
- 记忆：查询、确认、拒绝、纠正、删除和导出 Phase 3 受控记忆。

服务端环境变量：

- MEDAGENT_API_BASE，默认 http://127.0.0.1:8000
- MEDLIVE_API_BASE，默认 http://127.0.0.1:9821
- LIVERAG_API_BASE 仅保留为 MEDLIVE_API_BASE 的兼容别名

LiveKit 密钥和语音 provider 密钥不再放在浏览器前端环境中。

## 许可证与来源

- licenses/LiveRAG-MIT.txt 保存 LiveRAG MIT 许可证全文。
- frontend/LICENSE 保留前端上游许可证。
- THIRD_PARTY_NOTICES.md 记录来源、commit、工作区差异、迁入路径和无 Git 历史策略。

## 验证结果

Python 语法：

    python -m compileall -q src/medcontracts src/medlive src/medrag
    通过

Phase 3 前置：

    python -m pytest -q tests/test_phase3_memory.py tests/test_phase3_postgres_integration.py
    24 passed, 1 skipped

MedLive 与 Phase 4 回归（使用 D:\LiveRAG 锁定环境并临时提供 pytest-asyncio）：

    uv run --project D:\LiveRAG --with pytest-asyncio python -m pytest -q -o asyncio_mode=auto ...
    57 passed, 2 warnings

前端类型：

    frontend/node_modules/.bin/tsc.cmd --noEmit
    通过

前端生产构建：

    frontend/node_modules/.bin/next.cmd build
    通过；9 个 app routes 生成成功

构建仍报告 LiveRAG 上游已有的非阻断 lint warnings，包括 img 优化、少量未使用变量和 React Hook dependency 建议。

## 边界与未完成项

- 没有创建 Compose、Dockerfile、standalone 部署产物、profiles、数据卷、备份、健康检查或切换脚本。
- 没有处理 Phase 5 部署与运维。
- 没有迁移 LiveRAG SQLite、history 或个人知识库数据。
- 没有声称真实 LiveKit、STT、Voice LLM、TTS 或真实麦克风通过。
- 没有声称本轮真实 PostgreSQL 集成通过；Phase 3 PostgreSQL 测试在前置检查中跳过。
- Next.js 本地依赖安装遵循锁文件；供应链策略忽略的原生构建脚本没有被批准或执行，但直接 TypeScript 检查和 Next.js build 均成功。
- 原有 MedAgent 未提交修改与删除项保持在工作区，未清理、提交或回退。

## 后续验收建议

后续工作如需继续，应先由用户明确启动新的阶段。Phase 5 只能在独立授权下处理，并应从真实 PostgreSQL、LiveKit/provider 凭据和备份/回滚演练开始。
