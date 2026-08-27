# 演枢台（Node.js，零依赖）

只编排、只展示、不计算。当前为 **P1 骨架**：

- `GET /api/state`：平台状态；
- `GET /api/stream`：SSE 事件流（自动重连、回放最近 50 条）；
- `POST /api/nodes/probe`：探活 U/M/S 的 `/v1/hello`；
- `POST /api/session/start`：生成 `trace_id` 并发起 INIT 动作；
- `POST /api/reset`：复位；
- `/`：master 页骨架（后续扩展 party-u/m/s、eval 四页）。

运行：`node server/index.js`（默认 :8600）。
后续阶段（P2+）：阶段状态机（INIT→DATA→FORWARD→BACKWARD→RECONSTRUCT→UPDATE→EVAL）、
五页 UI、本地演示模式、L1 检查点切换。
