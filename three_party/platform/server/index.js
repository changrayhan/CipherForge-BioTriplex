#!/usr/bin/env node
/* 演枢台（零依赖）：状态机 + SSE 事件流 + 节点探活 + 会话发起 + 静态页。
 * P1 骨架：master 页可看模式/节点/会话；完整五页与阶段编排在 P2+。
 */
"use strict";

const http = require("http");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { URL } = require("url");

const ROOT = path.join(__dirname, "..");
const WEB = path.join(ROOT, "web");
const CONFIG = JSON.parse(
  fs.readFileSync(path.join(ROOT, "config", "platform.config.json"), "utf-8")
);
const PRESETS_PATH = path.join(ROOT, "..", "data", "fixtures", "eval-presets.json");
let WORKER_CONFIG = {};
try {
  WORKER_CONFIG = JSON.parse(
    fs.readFileSync(path.join(ROOT, "..", "coordinator", "three_party_config_rms.json"), "utf-8")
  );
} catch {}

const state = {
  mode: CONFIG.platform.mode,
  nodes: { U: { status: "unknown" }, M: { status: "unknown" }, S: { status: "unknown" } },
  session: null,
  eval: null,
  events: [],
};
const MAX_EVENTS = 800;
const subscribers = new Set();

function broadcast(ev) {
  ev = { id: state.events.length + 1, ts: Date.now(), ...ev };
  state.events.push(ev);
  if (state.events.length > MAX_EVENTS) state.events.splice(0, state.events.length - MAX_EVENTS);
  const line = `data: ${JSON.stringify(ev)}\n\n`;
  for (const res of subscribers) res.write(line);
}

function sendJson(res, obj, status = 200) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (c) => {
      raw += c;
      if (raw.length > 1 << 20) req.destroy();
    });
    req.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch {
        resolve({});
      }
    });
    req.on("error", () => resolve({}));
  });
}

function readPresets() {
  try {
    return JSON.parse(fs.readFileSync(PRESETS_PATH, "utf-8"));
  } catch {
    return { required: ["acc", "auprc", "macro_f1"], tests: {} };
  }
}

function nodeUrl(role) {
  return CONFIG.nodes[role] && CONFIG.nodes[role].url;
}

async function callNode(role, pathname, payload) {
  const url = nodeUrl(role);
  const timeout = (CONFIG.nodes[role] && CONFIG.nodes[role].timeout_ms) || 30000;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const resp = await fetch(url + pathname, {
      method: payload ? "POST" : "GET",
      headers: payload ? { "Content-Type": "application/json" } : undefined,
      body: payload ? JSON.stringify(payload) : undefined,
      signal: ctrl.signal,
    });
    return await resp.json();
  } finally {
    clearTimeout(timer);
  }
}

async function probe() {
  for (const role of ["U", "M", "S"]) {
    try {
      const h = await callNode(role, "/v1/hello", null);
      state.nodes[role].status = h.ok ? "online" : "error";
      state.nodes[role].info = { role: h.role, proto: h.protocol_version };
    } catch {
      state.nodes[role].status = "offline";
    }
  }
  broadcast({ type: "node", party: null, payload: { msg: "节点探活完成", nodes: state.nodes } });
}

async function startSession() {
  const trace_id =
    "task_" + new Date().toISOString().replace(/\D/g, "").slice(0, 14) +
    "_" + Math.random().toString(16).slice(2, 6);
  state.session = { trace_id, step: 0, stage: "INIT" };
  broadcast({ type: "session", party: null, trace_id, payload: { msg: "会话发起" } });
  // INIT 动作：M 出密钥 → S 建密文库 → S 登记 PRG → U 登记 PRG（docs/02 阶段-动作矩阵）
  const workerParams = {
    vocab_size: WORKER_CONFIG.vocab_size || 32000,
    hidden_dim: WORKER_CONFIG.hidden_dim || 2048,
    poly_degree: WORKER_CONFIG.poly_degree || 4096,
    plain_bits: WORKER_CONFIG.plain_bits || 30,
    scale: WORKER_CONFIG.scale || 10000,
    lam: WORKER_CONFIG.lam || 80,
  };
  const prg_seed_b64 = crypto.randomBytes(32).toString("base64");
  const initActions = [
    { role: "M", stage: "INIT", action: "bfv_keygen", params: workerParams },
    { role: "S", stage: "INIT", action: "build_enc_db", params: { ...workerParams } },
    { role: "S", stage: "INIT", action: "pir_prg_setup", params: { prg_seed_b64 } },
    { role: "U", stage: "INIT", action: "pir_prg_setup", params: { prg_seed_b64 } },
  ];
  let pk_pem_b64 = "";
  for (const a of initActions) {
    let params = a.params;
    if (a.action === "build_enc_db" && pk_pem_b64) {
      params = { ...params, pk_pem_b64 };
    }
    try {
      const r = await callNode(a.role, "/v1/action", {
        protocol_version: "1.0", trace_id, stage: a.stage, step: 0, action: a.action, params,
      });
      if (a.action === "bfv_keygen" && r.ok) {
        pk_pem_b64 = r.result && r.result.pk_pem_b64;
      }
      broadcast({
        type: r.ok ? "action_done" : "action_failed", party: a.role, trace_id,
        payload: { msg: `${a.action} ${r.ok ? "完成" : "失败"}`, code: r.error && r.error.code },
      });
    } catch (e) {
      broadcast({ type: "action_failed", party: a.role, trace_id, payload: { msg: String(e) } });
    }
  }
}

async function runFallback() {
  const ck = (CONFIG.fallback && CONFIG.fallback.pretrained_checkpoint_id) || "";
  const trace_id = state.session && state.session.trace_id || "";
  // docs/04 §2.2：平台依次调用 S → M → U
  const order = ["S", "M", "U"];
  const results = {};
  for (const role of order) {
    try {
      const r = await callNode(role, "/v1/fallback/pretrained", {
        protocol_version: "1.0", trace_id, checkpoint_id: ck,
      });
      results[role] = r;
      broadcast({
        type: r.ok ? "fallback_loaded" : "fallback_failed", party: role, trace_id,
        payload: {
          msg: r.ok ? `检查点 ${ck} 已加载` : `加载失败：${r.error && r.error.message}`,
          checkpoint_id: ck,
        },
      });
    } catch (e) {
      results[role] = { ok: false, error: { code: 0, message: String(e), retryable: false } };
      broadcast({ type: "fallback_failed", party: role, trace_id, payload: { msg: String(e) } });
    }
  }
  state.fallback = results;
  return results;
}

async function runEval(test_id) {
  const trace_id = state.session && state.session.trace_id || "";
  const r = await callNode("S", "/v1/eval/run", {
    protocol_version: "1.0", trace_id, test_id, params: {},
  });
  state.eval = { test_id, result: r };
  broadcast({
    type: r.ok ? "eval_done" : "eval_failed", party: "S", trace_id,
    payload: { msg: `${test_id} ${r.ok ? "完成" : "失败"}`, result: r.result },
  });
  return r;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const p = url.pathname;

  if (p === "/api/state") return sendJson(res, state);
  if (p === "/api/nodes/probe") { await probe(); return sendJson(res, state); }
  if (p === "/api/session/start") { await startSession(); return sendJson(res, state); }
  if (p === "/api/fallback/pretrained" && req.method === "POST") {
    const results = await runFallback();
    return sendJson(res, results);
  }
  if (p === "/api/eval/run" && req.method === "POST") {
    const body = await readBody(req);
    const test_id = String(body.test_id || "acc");
    const r = await runEval(test_id);
    return sendJson(res, r);
  }
  if (p === "/api/eval/tests") {
    return sendJson(res, readPresets());
  }
  if (p === "/api/ingest" && req.method === "POST") {
    const body = await readBody(req);
    broadcast({
      type: body.type || "ingest",
      party: body.party || null,
      trace_id: body.trace_id || null,
      payload: body.payload || body,
    });
    return sendJson(res, { ok: true });
  }
  if (p === "/api/reset") {
    state.session = null; state.eval = null; state.fallback = null; state.events = [];
    broadcast({ type: "state", party: null, payload: { msg: "平台复位" } });
    return sendJson(res, state);
  }
  if (p === "/api/stream") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    res.write(`data: ${JSON.stringify({ type: "hello", ts: Date.now(), payload: { msg: "已连接" } })}\n\n`);
    for (const ev of state.events.slice(-50)) res.write(`data: ${JSON.stringify(ev)}\n\n`);
    subscribers.add(res);
    req.on("close", () => subscribers.delete(res));
    return;
  }

  // static
  let file = path.join(WEB, p === "/" ? "index.html" : p);
  if (!file.startsWith(WEB)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end("not found"); }
    const ext = path.extname(file);
    const mime = { ".html": "text/html", ".js": "application/javascript", ".css": "text/css" };
    res.writeHead(200, { "Content-Type": `${mime[ext] || "text/plain"}; charset=utf-8` });
    res.end(data);
  });
});

server.listen(CONFIG.platform.port, () => {
  console.log(`[演枢台] listening on :${CONFIG.platform.port} (mode=${state.mode})`);
  probe();
});
