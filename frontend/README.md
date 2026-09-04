# MedAgent 前端

MedAgent 的统一 Web 前端，基于 Next.js 15、React 19 与 TypeScript。产品界面以中文医疗咨询为核心，实时语音由 MedLive/LiveKit 提供底层能力。

## 主要体验

- 登录后默认进入“医疗咨询”。
- 文字输入与语音输入位于同一个输入区，切换方式不会跳页或清空时间线。
- 文字消息、实时语音转写、助手回答和逐回答依据显示在同一条时间线。
- 病历/医学资料、历史会话、健康档案和设置是辅助功能。
- 回答依据依附于具体回答，默认折叠。
- 桌面端使用医疗工作台侧栏，移动端使用底部导航和安全区适配。

## 安全边界

浏览器不保存 JWT。登录和注册响应中的访问令牌由 Next.js 路由写入 HttpOnly Cookie：

- MedAgent 请求经同源 /api/medagent/\* 代理。
- MedLive 请求经同源 /api/liverag/\* 兼容代理。
- 语音 session 由 /api/token 使用 HttpOnly Cookie 在服务端创建。
- 浏览器端不能读取访问令牌或当前 voice session cookie。

请勿恢复旧版使用 localStorage 保存 JWT 的实现。frontend-legacy/ 只作为迁移参考和兼容入口。

## 统一会话模型

前端为每次可见咨询创建父 conversation_id：

    conversation_id: conv_<random>
    ├─ 文字子会话: web_conv_<random>
    └─ 语音子会话: vs_<server-generated>
       └─ MedLive client_id = conv_<random>

frontend/lib/medical-conversation.ts 负责合并文字消息、LiveKit 实时转写和 MedLive 持久化语音轮次，并优先保留带回答依据的持久化版本。

当前边界：

- 父 ID 已在两条管线中可关联，但后端仍以不同 session_id 保存文字和语音。
- 同一登录用户共享健康档案。
- 语音使用当前选中的病历/医学资料库；现有 MedAgent 文字接口的 knowledge_base 字段是医疗科室语义，不等同于 MedLive 资料库 ID，因此本次没有错误地把资料库 ID 注入文字接口。
- 因后端短期上下文仍按子会话隔离，文字模型不会自动看到刚刚的语音轮次，语音模型也不会自动看到本页文字轮次；界面将其表述为“同一咨询记录”，不宣称底层完整上下文已经互通。

要实现真正共享的短期上下文，后续应由后端增加正式的 parent conversation 契约，并在两个编排器生成前共同读取其规范化消息。

## 本地开发

安装依赖后运行 pnpm dev。

常用校验：

- pnpm typecheck
- pnpm test
- pnpm build

Windows 本地构建使用标准 Next.js 输出，避免普通账户无法创建 standalone 符号链接；Linux 与容器构建继续生成 standalone 输出。

## 关键文件

- components/app/unified-app.tsx：登录、医疗信息架构、响应式应用外壳。
- components/app/medical-consultation.tsx：统一时间线、文字流、语音控制和逐回答依据。
- lib/medical-conversation.ts：父会话标识、时间线聚合、模式与语音状态映射。
- app/api/token/route.ts：安全创建语音子会话并写入 HttpOnly session cookie。
- lib/medagent-api.ts：文字流、历史会话和健康档案 API。
