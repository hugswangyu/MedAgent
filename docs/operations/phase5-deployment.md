# Phase 5 部署与运维手册

## 安全边界

- voice 启动 PostgreSQL、迁移门禁、MedAgent API、LightRAG、MedLive API/worker、LiveKit 和 Next.js。
- full 在上述服务之外启动 Milvus（etcd/MinIO）、Elasticsearch 和 Neo4j。
- PostgreSQL 与医学数据服务不发布宿主端口；HTTP 用户入口只绑定 127.0.0.1。跨主机部署必须在受控反向代理、TLS、防火墙和独立 secret store 后显式修改。
- 仓库内 deploy/secrets/*.example 仅用于本机验证。共享或生产部署必须将所有 *_SECRET_FILE 指向仓库外、权限受限的独立文件。
- 默认禁止公开注册。生产必须使用 MEDRAG_ENV=prod；config-guard 会读取实际挂载的 Compose secrets，并拒绝仓库示例密钥、缺失密钥以及 Neo4j/MinIO 默认开发凭据。门禁通过后才允许数据和核心服务启动。

## 配置与启动

复制 deploy/.env.phase5.example 为被 Git 忽略的 deploy/.env.phase5，填写 provider 配置；不要把真实密钥写回示例文件。

    docker compose --env-file deploy/.env.phase5 --profile voice config --quiet
    docker compose --env-file deploy/.env.phase5 --profile voice up -d --build

完整医学环境：

    docker compose --env-file deploy/.env.phase5 --profile full config --quiet
    docker compose --env-file deploy/.env.phase5 --profile full up -d --build

Python 镜像按用途拆为 ops-runtime、medical-runtime 和 live-runtime。medical/live 固定从 PyTorch 官方 CPU 索引安装 torch，不包含 CUDA runtime；pip 下载超时为 300 秒并最多重试 10 次。修改依赖后应在镜像内确认 torch.version.cuda 为 None。

查看门禁与健康状态：

    docker compose --profile voice ps
    docker compose --profile voice logs migrate medagent-api lightrag medlive-api medlive-worker

## 切换前门禁

先启动 PostgreSQL，再在事务中执行全部待处理 SQL 并回滚。checksum 不一致会硬失败：

    docker compose --profile voice up -d postgres
    docker compose --profile voice run --rm migrate python /app/deploy/ops/migrate.py dry-run

随后做备份，正式应用迁移并启动。Compose 中的 migrate 是 API 的 service_completed_successfully 前置依赖，迁移失败时 API 不会启动：

    pwsh -File deploy/ops/backup.ps1 -Profile voice
    docker compose --profile voice up -d

从已加载 secrets 的 MedLive 容器执行预热与 smoke，避免发布 LightRAG 内部端口：

    docker compose --profile voice exec medlive-api python /app/deploy/ops/http_checks.py prewarm --medagent http://medagent-api:8000 --medlive http://medlive-api:9821 --lightrag http://lightrag:9721
    docker compose --profile voice exec medlive-api python /app/deploy/ops/http_checks.py smoke --medagent http://medagent-api:8000 --medlive http://medlive-api:9821 --lightrag http://lightrag:9721

没有 provider 密钥的本机配置只能用 --allow-provider-missing 验证进程存活，不能作为部署切换验收。指定 --kb-id 会额外加载目标知识库以完成预热。

## 数据卷与备份

事实源为 postgres-data；个人库、LightRAG SQLite、上传和派生索引位于 medlive-data；MedAgent 本地运行态位于 medagent-data。完整环境另有 Milvus/etcd/MinIO、Elasticsearch、Neo4j 数据卷。

backup.ps1 只停止备份开始时正在运行的写入者，再执行 PostgreSQL custom-format 逻辑备份和冷卷快照，最后生成包含 SHA-256 的 backups/<UTC timestamp>/manifest.json。finally 只恢复这些被脚本停止的写入者，不会启动原先停止的服务。BackupRoot 必须精确等于仓库 backups 目录；同名前缀目录、路径逃逸和 reparse point 均会被拒绝。备份目录已被 Git 忽略；应由外部备份系统加密、复制并按保留策略清理。

    pwsh -File deploy/ops/backup.ps1 -Profile full -ProjectName medagent

恢复会覆盖当前卷，必须在隔离环境先演练，并显式传入确认开关：

    pwsh -File deploy/ops/restore.ps1 -BackupId 20260902T120000123Z-a1b2c3d4e5f6 -Profile full -ProjectName medagent -ConfirmRestore

BackupId 使用 UTC 毫秒时间戳加 12 位随机十六进制后缀（例如 20260902T120000123Z-a1b2c3d4e5f6），创建时若目标目录已存在会直接失败，绝不复用。恢复脚本要求备份目录是 backups 的直接子目录，并严格校验 schema 为整数 2、manifest 文件名唯一、profile 对应的 payload 集合完整且与磁盘集合完全一致、文件大小和 SHA-256；同时拒绝 reparse point、非白名单文件名和路径逃逸。随后记录原运行服务集合、停止该集合、恢复冷卷和 PostgreSQL。结束时会停止恢复期间临时启动的服务，并用 --no-deps 精确恢复原集合；原集合为空时恢复后仍保持全停。恢复后必须重新运行迁移 dry-run、预热、smoke 和租户隔离测试。

## 回滚演练

每次切换记录旧 Git ref、Compose 渲染结果、镜像 digest、备份 ID 和迁移 checksum。演练顺序：

1. 用 backup.ps1 创建切换前备份，复制到隔离存储并验证 manifest。
2. 运行迁移 dry-run；失败则不切换。
3. 部署新版本，等待所有 healthcheck，通过预热和 smoke。
4. 若应用异常但 schema 兼容，停止新栈，用旧 Git ref/旧镜像和原 Compose 配置重启。
5. 若数据或 schema 异常，停止全栈，使用切换前备份执行 restore.ps1 -ConfirmRestore，再部署旧版本。
6. 重新验证用户隔离、语音绑定防重放、未检查文本不进入 TTS、重复 finalize 幂等。

禁止只回退代码却保留未经验证的新 schema，也禁止在仍有写入者时直接复制数据库、Milvus、Elasticsearch 或 Neo4j 卷。

## 导出与删除

- 用户记忆导出使用 GET /memories/export；确认、拒绝、纠正和删除继续走 Phase 3 API。
- 个人知识库及文档通过 MedLive /rag/knowledge-bases/* API 导出或删除；先由 PostgreSQL 所有权校验，再操作 LightRAG。
- 删除知识库时保留不可还原的最小审计标识，清除证据正文、上传原件和派生内容；执行后检查 PostgreSQL 状态机最终为 deleted。
- PostgreSQL dump 和卷备份可能包含已删除前数据，必须按数据保留期到期销毁相应备份，并记录操作审计。

## 生产补充要求

当前 Compose 是单机基线，不包含公网 TLS、集中日志、告警、异地备份、LiveKit TURN 或 provider 凭据。公网/多机上线前必须补齐这些能力，并将入口从 localhost 改到受控代理网络；不得直接发布数据库端口。
