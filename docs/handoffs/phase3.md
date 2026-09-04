# MedAgent × LiveRAG Phase 3 前置交接核查

日期：2026-09-02
状态：离线定向验收通过；真实 PostgreSQL 集成未在 Phase 4 前置检查中复验
提交：`9449c72`（Phase 3 受控记忆闭环）；以下内容在 Phase 4 开始前已存在于工作区

## 核查结论

Phase 3 没有独立交接文件。Phase 4 开始时发现以下未提交实现：

- scripts/phase3_memory.sql
- src/medrag/app/api/memories.py
- src/medrag/memory/controlled.py
- tests/test_phase3_memory.py
- tests/test_phase3_postgres_integration.py
- 与消息、会话、仓储和服务相关的现有修改

Phase 4 没有覆盖或回退这些内容，只把它们作为迁移基线读取和测试。

## 前置验证

主仓 Python 环境：

    python -m pytest -q tests/test_phase3_memory.py tests/test_phase3_postgres_integration.py
    24 passed, 1 skipped in 38.88s

跳过项是需要真实 PostgreSQL 的集成验收。因本次任务边界是 Phase 4 迁仓与前端，不启动或部署 Phase 5 基础设施，因此没有把该跳过项描述为通过。

## Phase 4 准入判断

离线记忆闭环、幂等 finalize、受控事实状态变更、替代链、所有权隔离和导出接口可作为 Phase 4 前端接入基线。真实 PostgreSQL 与真实语音 provider 验收仍保持外部限制。
