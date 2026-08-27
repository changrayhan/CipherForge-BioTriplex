#!/usr/bin/env node
/* demo/nodes/index.js —— 文档 00 FAQ Q1 引用的 Mock 节点实现。
 *
 * 模拟全部协议动作（docs/02 阶段-动作矩阵），返回合成事件/指标，
 * 供本地演示（L2）与联调对照使用。真实模式请使用 party_u/m/s 服务。
 */
"use strict";

const http = require("http");

const ROLE = process.env.CF_MOCK_ROLE || "U";
const PORT = Number(process.env.CF_MOCK_PORT || 9100 + (ROLE === "U" ? 1 : ROLE === "M" ? 2 : 3));

const ACTIONS = [
  "bfv_keygen", "build_enc_db", "pir_prg_setup", "load_dataset", "register_task",
  "local_embed", "trunk_forward", "head_forward", "pir_query_mask", "fetch_rows",
  "share_compute", "rms_parity", "grad_reconstruct", "lora_update", "open_eval",
];

function json(res, obj, status = 200) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(body);
}

function readBody(req, cb) {
  let raw = "";
  req.on("data", (c) => { raw += c; });
  req.on("end", () => {
    try { cb(raw ? JSON.parse(raw) : {}); } catch { cb({}); }
  });
}

const server = http.createServer((req, res) => {
  const p = req.url.split("?")[0].replace(/\/+$/, "") || "/";
  if (req.method === "GET" && p === "/v1/hello") {
    return json(res, {
      ok: true, node_id: `mock-${ROLE.toLowerCase()}-01`, role: ROLE,
      protocol_version: "1.0",
      capabilities: ["action", "fallback", "eval"],
      assets: [`Mock ${ROLE} 资产`],
      status: "idle",
    });
  }
  if (req.method === "POST" && p === "/v1/action") {
    return readBody(req, (b) => {
      const action = b.action || "unknown";
      json(res, {
        ok: ACTIONS.includes(action),
        events: [{ type: "message", payload: { msg: `[Mock ${ROLE}] ${action}`, edge: `${ROLE}->*`, bytes: 1024 } }],
        metrics: { comm: [{ edge: `${ROLE}->*`, bytes: 1024 }], step_ms: 10 },
        result: { status: "success", action },
        error: ACTIONS.includes(action) ? undefined : { code: 501, message: "not implemented", retryable: false },
      });
    });
  }
  if (req.method === "POST" && p === "/v1/fallback/pretrained") {
    return readBody(req, (b) => json(res, {
      ok: true, checkpoint: b.checkpoint_id || "mock-checkpoint", loaded: true, load_ms: 1,
    }));
  }
  if (req.method === "POST" && p === "/v1/eval/run") {
    return readBody(req, (b) => json(res, {
      ok: true,
      result: {
        status: "DONE",
        metrics: [
          { key: "before", label: "微调前", value: 61.2, unit: "%" },
          { key: "after", label: "微调后", value: 88.5, unit: "%" },
        ],
        conclusion: `[Mock] ${b.test_id || "acc"} 预设回放`,
      },
    }));
  }
  json(res, { ok: false, error: { code: 404, message: `not found: ${req.url}`, retryable: false } }, 404);
});

server.listen(PORT, () => {
  console.log(`[mock ${ROLE}] listening on :${PORT}`);
});
