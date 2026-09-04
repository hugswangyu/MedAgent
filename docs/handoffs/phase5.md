# MedAgent × LiveRAG Phase 5 交接记录

日期：2026-09-04
阶段：部署与运维
状态：Go；Phase 5 已通过离线、容器构建与隔离 full 备份恢复验收；真实语音 provider 与正式环境上线验收仍待生产凭据
提交：`07b72f4`（Phase 5 部署与运维）

## Phase 4 前置状态

Phase 4 交接文件已核对。迁仓、medcontracts、统一 Next.js、许可证与离线回归均已完成；当时明确没有 Compose、profiles、卷、备份、健康检查和切换脚本。真实 PostgreSQL、LiveKit、STT/LLM/TTS provider 端到端因无凭据未验收。本阶段没有声称这些外部能力已经通过。

开始 Phase 5 时已有以下用户工作区内容，本阶段全部保留且未编辑：docs/handoffs/phase1.md 的修改，三项已删除文档，以及 fastapi-startup.err / fastapi-startup.log。

## 完成范围

1. 新增 compose.yaml，统一编排 MedAgent API、MedLive API/worker、LiveKit、LightRAG、Next.js、PostgreSQL、Milvus、Elasticsearch 和 Neo4j；Milvus 的 etcd/MinIO 依赖也纳入同一编排。
2. voice profile 包含配置门禁和最小语音闭环的 9 个服务；full profile 在其上增加 5 个完整医学检索服务。
3. 数据服务仅在 Compose 私有网络可见；用户入口只绑定本机 127.0.0.1。默认关闭公开注册，Elasticsearch 开启密码鉴权。
4. 新增 Python 与 Next.js 多阶段 Dockerfile，Next.js 使用 standalone 输出；Python 分为 ops、medical、live target，医学与语音镜像固定使用官方 CPU Torch。
5. 为 PostgreSQL、MedAgent、MedLive/LightRAG、Milvus、Elasticsearch 和 Neo4j 建立具名持久卷；PostgreSQL 同时挂载被 Git 忽略的备份目录。
6. 应用敏感值通过 Compose file secrets 注入；仓库只包含明确标注的本机示例值，不含真实生产密钥。生产 config-guard 读取实际挂载值，拒绝缺失/示例密钥和 Neo4j/MinIO 默认开发凭据。
7. 所有长期运行服务配置 healthcheck。API 在迁移服务成功退出后启动；MedLive API 等待 MedAgent 与 LightRAG 存活，worker 等待 LiveKit 与 MedLive API。
8. deploy/ops/migrate.py 对 4 份 SQL 按顺序执行，使用 advisory transaction lock、SHA-256 checksum 和 schema_migrations；dry-run 会实际执行 SQL 后回滚。
9. deploy/ops/http_checks.py 提供 smoke 与预热；可预热指定知识库，缺少 provider 时默认失败，只有显式 allow-provider-missing 才允许本机存活检查。
10. backup.ps1 严格限制 BackupRoot 为仓库 backups，使用 UTC 毫秒时间戳加随机后缀；目录预留先在同一根目录用 FileMode.CreateNew 与 FileShare.None 取得独占声明，持锁创建目标目录，且只有锁所有者清理声明文件，因此并发相同 BackupId 只能有一个成功者。备份只恢复原先运行并被停止的写入者；restore.ps1 严格校验 BackupId、manifest schema、文件名唯一性、完整文件集合、直接子目录、reparse point、大小和 SHA-256，覆盖恢复后精确还原原服务运行集合。
11. docs/operations/phase5-deployment.md 记录启动、密钥、迁移 dry-run、预热、备份恢复、回滚演练、导出删除与生产补充要求。

## Profile 服务集合

voice:

- config-guard, postgres, migrate, medagent-api, lightrag
- medlive-api, livekit, medlive-worker, frontend

full:

- voice 的全部服务
- etcd, minio, milvus, elasticsearch, neo4j

## 验证结果

Compose 解析：

    docker compose --profile voice config --quiet
    通过

    docker compose --profile full config --quiet
    通过

profile 服务清单已逐项核对，voice 为 9 个服务，full 为 14 个服务。

Python 语法：

    python -m compileall -q deploy/ops src/medcontracts src/medlive src/medrag
    通过

PowerShell 语法：

    backup_contract.ps1、backup.ps1 与 restore.ps1 使用 PowerShell Parser 解析
    通过

Phase 5 与 Phase 4 定向测试：

    python -m pytest -q tests/test_phase5_deployment.py tests/test_phase4_migration.py
    30 passed

其中新增测试会实际执行 PowerShell：生成两份备份预留并验证 ID 唯一、拒绝已存在目录；12 个独立 PowerShell 进程经同一 gate 同时争抢相同 BackupId，结果严格为 1 个成功、11 个失败、1 个目标目录且无残留 .reserve 文件。manifest 合法集合与错误/字符串 schema、重复文件名、缺失 payload、磁盘额外文件均参数化覆盖 voice 和 full；另验证 BackupRoot 同名前缀目录和路径型 BackupId。测试不再用源码关键字断言代替这些行为。

Phase 0–5 定向回归：

    77 passed, 1 skipped, 1 warning

跳过项仍为需要真实 PostgreSQL 的 Phase 3 集成测试。warning 是 FastAPI TestClient 的上游弃用提示。

前端 TypeScript：

    frontend/node_modules/.bin/tsc.cmd --noEmit
    通过（在 frontend 工作目录）

Git 空白检查：

    git diff --check
    通过

前一轮镜像构建与 CPU 校验基线：

    docker build -t medagent-frontend:phase5-acceptance -f deploy/Dockerfile.frontend .
    docker build --target ops-runtime -t medagent-ops:phase5-acceptance -f deploy/Dockerfile.python .
    docker build --target medical-runtime -t medagent-medical:phase5-acceptance -f deploy/Dockerfile.python .
    docker build --target live-runtime -t medagent-live:phase5-acceptance -f deploy/Dockerfile.python .

四个 target 在前一轮均构建成功，medical/live 容器内均为 torch 2.8.0+cpu，torch.version.cuda 为 None。本轮随后为 Dockerfile 增加 PIP_DEFAULT_TIMEOUT=300 和 PIP_RETRIES=10；依照本次验收意见，没有重复镜像构建。本机 BuildKit 曾临时使用 --add-host 处理官方 CDN DNS，该地址未写入仓库。

生产配置门禁：开发示例配置在 dev 模式可用；同一组仓库示例 secrets 在 MEDRAG_ENV=prod 时被一次性 config-guard 容器逐项拒绝。

隔离备份恢复演练：

- project：phase5acceptance0902，仅启动 config-guard/postgres，并使用独立 postgres-data、medagent-data、medlive-data 卷。
- PostgreSQL 表及两个应用卷写入 before 哨兵，执行 voice 备份后改为 after，再执行 ConfirmRestore；数据库和两个卷均恢复为 before。
- 原运行集合只有 postgres 时，恢复后仍只有 postgres；原运行集合为空时，临时 postgres 在恢复后停止，最终集合仍为空。
- 演练结束已删除该 project、网络、三个隔离卷和测试备份目录；未触碰原有 medagent-phase3-go-pg。

full profile 隔离恢复演练（2026-09-04 实际执行）：

- 唯一 project 为 phase5fullaccept0904a；启动前确认同名容器和 8 个同名前缀卷均不存在。只启动该 project 的 postgres 与 minio，另外为 full 合同创建 medagent-data、medlive-data、milvus-etcd、milvus-minio、milvus-data、elasticsearch-data、neo4j-data 独立卷。命令进程内清空 provider API key，数据服务只使用仓库示例开发凭据，没有使用真实凭据。
- 实际启动及运行集合记录：

      docker compose --project-name phase5fullaccept0904a --profile full up -d --no-deps postgres minio
      docker compose --project-name phase5fullaccept0904a --profile full ps --status running --services
      minio
      postgres

- PostgreSQL 的 phase5_acceptance 表及 7 个 full 冷卷分别写入可区分的 before 哨兵。实际备份命令与结果：

      .\deploy\ops\backup.ps1 -Profile full -ProjectName phase5fullaccept0904a
      Backup completed: D:\MedAgent\backups\20260904T022715927Z-de9f0177fbae
      schema=2, profile=full, project=phase5fullaccept0904a, files=8
      running_services_before_backup=minio,postgres
      running_services_after_backup=minio,postgres

- 备份后 PostgreSQL 和全部 7 个卷均改写为 after 并逐项读回确认。实际恢复命令：

      .\deploy\ops\restore.ps1 -BackupId 20260904T022715927Z-de9f0177fbae -Profile full -ProjectName phase5fullaccept0904a -ConfirmRestore

- 恢复结果：PostgreSQL 值回到 before；medagent-data、medlive-data、milvus-etcd、milvus-minio、milvus-data、elasticsearch-data、neo4j-data 均回到各自 before 哨兵；恢复后的运行服务集合仍严格为 minio、postgres，没有启动原先未运行的服务。
- 清理使用精确 project 名；备份目录在 Resolve-Path 后验证父目录严格等于 D:\MedAgent\backups 才递归删除：

      docker compose --project-name phase5fullaccept0904a --profile full down --volumes --remove-orphans
      CLEANUP_CONTAINERS=0
      CLEANUP_VOLUMES=0
      CLEANUP_BACKUP_EXISTS=False
      PROTECTED_VOLUMES_MISSING=0

  8 个隔离卷均已删除；本次 BackupId 目录已删除。既有 medagent_* 与 final_* 保护卷均仍存在，未操作现有数据。
  Docker Desktop 在演练前原为关闭状态；演练结束后已执行 DockerCli.exe -Shutdown，恢复宿主机原状态。

## 未执行与待验收

- 没有真实 provider 密钥，未运行真实 STT、voice LLM、TTS、麦克风、LiveKit 房间或 LightRAG provider-ready 验收。
- 没有对用户现有数据执行备份/恢复、迁移 apply 或破坏性回滚；本轮只操作并清理了隔离 project/卷。
- 单机 Compose 尚未包含公网 TLS、反向代理、TURN、集中日志/指标告警、异地加密备份和多机高可用。

## 下一步切换门禁

1. 将所有 secret file 路径切换到仓库外并轮换，配置真实 provider，设置 MEDRAG_ENV=prod。
2. 使用已验证的 CPU 镜像启动 PostgreSQL，运行 migrate.py dry-run。
3. 启动目标 profile，确认全部 healthy，执行 provider-ready 预热与 smoke。
4. 重跑真实 PostgreSQL、租户隔离、绑定防重放、TTS 输出闸门和 finalize 幂等测试。
5. 保存旧 Git ref、镜像 digest、Compose 渲染结果、备份 ID 与迁移 checksum 后再切换流量。
