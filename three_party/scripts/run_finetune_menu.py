#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三模式微调交互式启动器（明文LoRA基准 / 三进程RMS-PIR / 三进程Block-PIR）。

运行方式：
    bash three_party/scripts/run_finetune_menu.sh [--dry-run]

可选环境变量（非交互覆盖，便于冒烟测试）：
    CF_MODE         1|2|3|4
    CF_EPOCHS       默认 3
    CF_BATCH_SIZE   默认 16
    CF_OUT_ROOT     输出根目录（默认 /root/CipherForge/final-test-data）
    CF_MAX_STEPS    短跑步数（三进程 -> --max_train_steps，明文 -> --max_steps）
    CF_CRYPTO_ROOT  加密产物缓存根目录（默认 /root/autodl-tmp/CipherForge-final-test/crypto）
    CF_CACHE_CRYPTO 1=复用缓存（默认 0=每次重新生成 M密钥/S密文库/U hints）
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False


ROOT = Path(__file__).resolve().parents[1]            # three_party/
sys.path.insert(0, str(ROOT))
try:
    from coordinator.task_profiles import TASK_PROFILES, get_profile, DEFAULT_TASK
except Exception:  # pragma: no cover - fallback keeps the menu importable
    TASK_PROFILES = {"clinvar": {"label": "ClinVar 致病性二分类", "data_dir": str(ROOT / "party_u" / "data" / "qa"),
                                 "max_seq_length": 128, "dp_num_classes": 2,
                                 "class_outputs": ["Yes", "No"], "eval_mode": "binary"}}
    get_profile = lambda t: TASK_PROFILES.get(t, TASK_PROFILES["clinvar"])
    DEFAULT_TASK = "clinvar"
PY = "/root/miniconda3/bin/python3.12"
SNAP = "/root/hf_cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
QA_DIR = ROOT / "party_u" / "data" / "qa"
DEFAULT_OUT_ROOT = "/root/CipherForge/final-test-data"
DEFAULT_CRYPTO_ROOT = "/root/autodl-tmp/CipherForge-final-test/crypto"

MODES = {
    1: {
        "tag": "plaintext-baseline",
        "label": "明文LoRA基准",
        "kind": "plain",
        "config": None,
        "pir": "无加密/无DP",
        "dp": "无",
    },
    2: {
        "tag": "three-party-rms-pir",
        "label": "三进程RMS-PIR",
        "kind": "three",
        "config": "three_party_config_rms.json",
        "pir": "RMS-PIR（partition=200, lam=16）",
        "dp": "dp_enable=true, dp_alpha=0.03, dp_eta0=1500.0, dp_answer_beta=0.5, dp_num_classes=2",
    },
    3: {
        "tag": "three-party-block-pir",
        "label": "三进程Block-PIR",
        "kind": "three",
        "config": "three_party_config_block_dp.json",
        "pir": "Block-PIR（block=64, fake_ratio=0.25）",
        "dp": "dp_enable=true, dp_alpha=0.03, dp_eta0=1500.0, dp_answer_beta=0.5, dp_num_classes=2",
    },
}

DEFAULT_PARAMS = {
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "max_seq_len": 128,
    "seed": 42,
    "epochs": 3,
    "batch_size": 16,
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def banner(dataset: str = DEFAULT_TASK) -> None:
    p = get_profile(dataset)
    line = "=" * 78
    print(line)
    print("三模式隐私保护微调启动器（CipherForge）")
    print(line)
    print("当前微调模型（不可修改，仅提示）：")
    print(f"  模型名 : TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"  快照   : {SNAP}")
    print("当前微调数据集/任务（运行前可选择）：")
    print(f"  任务类型 : {dataset}")
    print(f"  数据集   : {p.get('label', dataset)}")
    print(f"  目录     : {p.get('data_dir', QA_DIR)}")
    print(f"  max_seq_len={p.get('max_seq_length', 128)}  "
          f"classes={len(p.get('class_outputs', []))}  eval={p.get('eval_mode', 'binary')}")
    print(line)


def ask_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            print(f"  输入无效，请输入 {lo}-{hi} 之间的整数")
            continue
        if lo <= v <= hi:
            return v
        print(f"  输入无效，请输入 {lo}-{hi} 之间的整数")


def choose_dataset() -> str:
    env_ds = _env("CF_DATASET")
    names = list(TASK_PROFILES.keys())
    if env_ds:
        if env_ds in TASK_PROFILES:
            return env_ds
        if env_ds.isdigit() and 1 <= int(env_ds) <= len(names):
            return names[int(env_ds) - 1]
        print(f"[警告] CF_DATASET={env_ds} 无效，取默认 {DEFAULT_TASK}")
    print("\n请选择微调数据集/任务类型（运行前选择）：")
    for i, name in enumerate(names, 1):
        p = TASK_PROFILES[name]
        print(f"  {i}) {p['label']}  [max_seq_len={p['max_seq_length']}, "
              f"classes={len(p['class_outputs'])}, eval={p['eval_mode']}]")
    while True:
        raw = input(f"请输入编号 [1-{len(names)}，回车默认 {DEFAULT_TASK}]：").strip()
        if not raw:
            return DEFAULT_TASK
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        if raw in TASK_PROFILES:
            return raw
        print("  输入无效")


def _count_jsonl_rows(data_dir: str) -> int:
    try:
        p = Path(data_dir) / "train.jsonl"
        if not p.exists():
            return 0
        n = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    except Exception:
        return 0


def choose_modes() -> list:
    env_mode = _env("CF_MODE")
    if env_mode:
        v = int(env_mode)
        if v == 4:
            return [1, 2, 3]
        if v in MODES:
            return [v]
        print(f"[错误] CF_MODE={env_mode} 无效，取默认 1")
    print("请选择微调模式：")
    print("  1) 明文 LoRA 基准（无加密、无 DP）")
    print("  2) 三进程 RMS-PIR 变体")
    print("  3) 三进程 Block-PIR 变体")
    print("  4) 按顺序全部执行（明文 → RMS → Block）")
    while True:
        raw = input("请输入模式编号 [1-4，回车默认 1]：").strip()
        if not raw:
            return [1]
        try:
            v = int(raw)
        except ValueError:
            print("  输入无效")
            continue
        if v == 4:
            return [1, 2, 3]
        if v in MODES:
            return [v]
        print("  输入无效，请输入 1-4")


def edit_params() -> dict:
    p = dict(DEFAULT_PARAMS)
    env_e = _env("CF_EPOCHS")
    env_b = _env("CF_BATCH_SIZE")
    if env_e:
        p["epochs"] = max(1, int(env_e))
    if env_b:
        p["batch_size"] = max(1, int(env_b))
    if not (env_e and env_b):
        print("\n默认微调参数（除 epoch 与 batch_size 外均不可修改）：")
        rows = [
            ("learning_rate", p["learning_rate"], "2e-4"),
            ("weight_decay", p["weight_decay"], "0.01"),
            ("warmup_steps", p["warmup_steps"], "100"),
            ("lora_rank / lora_alpha / lora_dropout", p["lora_rank"], "8 / 16 / 0.05"),
            ("max_seq_len", p["max_seq_len"], "128"),
            ("seed", p["seed"], "42"),
        ]
        for name, val, disp in rows:
            print(f"  {name:<40} {disp}")
        print("  （DP/PIR 参数按模式显示，见下方说明）")
        if not env_e:
            p["epochs"] = ask_int(f"可修改：epochs [1-20，回车默认 {p['epochs']}]：", p["epochs"], 1, 20)
        if not env_b:
            p["batch_size"] = ask_int(f"可修改：batch_size [1-64，回车默认 {p['batch_size']}]：", p["batch_size"], 1, 64)
    return p


def show_mode_params(mode: dict, dataset: str = DEFAULT_TASK) -> None:
    profile = get_profile(dataset)
    dp_desc = ("dp_enable=true, dp_alpha=0.03, dp_eta0=1500.0, dp_answer_beta=0.5, "
               f"dp_num_classes={profile.get('dp_num_classes', 2)}")
    print("\n" + "-" * 78)
    print(f"模式：{mode['label']}（{mode['tag']}）")
    print(f"  PIR 方案  : {mode['pir']}")
    print(f"  DP 噪声   : {dp_desc}")
    print(f"  说明      : 每次运行默认重新生成 M 方 BFV 密钥对、S 方密文库、U 方 hints")
    print(f"              （密文库与 hints 缓存于 /root/autodl-tmp；设 CF_CACHE_CRYPTO=1 可复用）")
    print("-" * 78)


def choose_out_root() -> str:
    env_root = _env("CF_OUT_ROOT")
    if env_root:
        return env_root
    while True:
        raw = input(f"\n请选择数据/日志保存根目录（回车默认 {DEFAULT_OUT_ROOT}）：").strip()
        path = raw or DEFAULT_OUT_ROOT
        if not os.path.isabs(path):
            print("  请输入绝对路径")
            continue
        return path


def run_dir_for(out_root: str, mode: dict) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path(out_root) / f"{mode['tag']}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def disk_warning() -> None:
    try:
        root_free = shutil.disk_usage("/").free
        tmp_free = shutil.disk_usage("/root/autodl-tmp").free
        if root_free < 1 << 30:
            print(f"[警告] 根盘剩余 {root_free / 1e9:.1f} GB < 1GB，产物可能写不进去")
        if tmp_free < 10 << 30:
            print(f"[警告] /root/autodl-tmp 剩余 {tmp_free / 1e9:.1f} GB，加密产物（约 4.2GB）空间紧张")
    except Exception:
        pass


def prepare_crypto(mode: dict, run_name: str) -> dict:
    """默认重新生成；CF_CACHE_CRYPTO=1 时复用共享缓存。返回各子目录。"""
    crypto_root = Path(_env("CF_CRYPTO_ROOT", DEFAULT_CRYPTO_ROOT))
    if _env("CF_CACHE_CRYPTO") == "1":
        base = crypto_root / "shared_cache" / mode["tag"]
        regenerate = False
    else:
        base = crypto_root / mode["tag"]
        regenerate = True
    dirs = {
        "m_keys": base / "m_keys",
        "enc_db": base / "enc_db",
        "rms_db": base / "rms_db",
        "rms_hints": base / "rms_hints",
    }
    if regenerate:
        if base.exists():
            shutil.rmtree(base)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"  加密产物目录（{'重新生成' if regenerate else '复用缓存'}）: {base}")
    return dirs


def kill_stale_nodes() -> None:
    for pat in ("party_u/main_u.py", "party_m/main_m.py", "party_s/main_s.py",
                "server/index.js"):
        subprocess.run(["pkill", "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    for port in (9001, 9002, 9003, 8600):
        for _ in range(20):
            try:
                with socket_create_connection(port):
                    pass
            except OSError:
                break
            time.sleep(0.5)


def socket_create_connection(port: int):
    import socket
    s = socket.create_connection(("127.0.0.1", port), timeout=0.3)
    s.close()
    return s


def port_open(port: int) -> bool:
    try:
        socket_create_connection(port)
        return True
    except OSError:
        return False


def wait_hello(port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/hello", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def wait_url(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def base_env() -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "HF_HOME": "/root/hf_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "MALLOC_ARENA_MAX": "2",
        "PYTHONMALLOC": "malloc",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


def stream_process(proc: subprocess.Popen, log_path: Path, line_handler) -> int:
    """逐行流式输出：原始行写 log 并打印，交给 handler 做增强展示。"""
    with log_path.open("a", encoding="utf-8") as logf:
        for raw in proc.stdout:
            text = raw.decode("utf-8", "replace").rstrip("\n")
            logf.write(text + "\n")
            logf.flush()
            line_handler(text)
    proc.wait()
    return proc.returncode


# --------------------------------------------------------------------------- #
#  明文模式
# --------------------------------------------------------------------------- #
PLAIN_DICT_RE = re.compile(r"\{[^{}]*\}")


def _hf_dict_from_line(line: str):
    """HF Trainer 的日志字典（单引号、字符串值）-> dict[float]。"""
    m = PLAIN_DICT_RE.search(line)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(0))
    except Exception:
        return None
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def run_plain(run_dir: Path, params: dict, max_steps: int, dry_run: bool,
              dataset: str = DEFAULT_TASK) -> dict:
    profile = get_profile(dataset)
    data_dir = Path(profile.get("data_dir", str(QA_DIR)))
    max_seq_len = int(profile.get("max_seq_length", params["max_seq_len"]))
    class_outputs = list(profile.get("class_outputs", ["Yes", "No"]))
    print(f"\n[明文] 输出目录: {run_dir}")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), "-u", "-s",
        str(ROOT.parent / "single_process" / "baseline" / "scripts" / "finetune_plain.py"),
        "--model_id", SNAP,
        "--data_dir", str(data_dir),
        "--out_dir", str(run_dir),
        "--max_seq_len", str(max_seq_len),
        "--epochs", str(params["epochs"]),
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["learning_rate"]),
        "--lora_r", str(params["lora_rank"]),
        "--lora_alpha", str(params["lora_alpha"]),
        "--lora_dropout", str(params["lora_dropout"]),
        "--seed", str(params["seed"]),
    ]
    if max_steps:
        cmd += ["--max_steps", str(max_steps)]
    eval_steps = int(_env("CF_EVAL_STEPS")) if _env("CF_EVAL_STEPS") else 0
    if eval_steps > 0:
        # Epoch-aligned eval/checkpoints so per-epoch metrics can be recorded.
        cmd += ["--eval_steps", str(eval_steps), "--save_total_limit", "6"]
    console = run_dir / "console.log"
    print("[\u660e\u6587] \u542f\u52a8\u8bad\u7ec3\uff1a" + " ".join(cmd))
    if dry_run:
        print("[DRY-RUN] 停止于训练启动前")
        return {"dry_run": True, "run_dir": str(run_dir)}

    n_train = _count_jsonl_rows(str(data_dir))
    steps_per_epoch = max(1, math.ceil(n_train / params["batch_size"])) if n_train else max(1, math.ceil(10000 / params["batch_size"]))
    total = max_steps if max_steps else int(steps_per_epoch * params["epochs"])
    state = {"step": 0, "loss": None, "first_loss": None, "min_loss": None, "eval_loss": None,
             "epoch": 0.0, "lr": None, "count": 0}

    def handler(line: str) -> None:
        d = _hf_dict_from_line(line)
        if d is None:
            return
        if d.get("train_runtime") is not None:
            # 训练结束汇总字典：补齐步数显示（tqdm 行无 JSON 字典可解析）
            state["step"] = max(state["step"], total)
            return
        if d.get("eval_loss") is not None:
            # 验证字典只更新 eval_loss，不推进 step（否则 step 会虚高）
            state["eval_loss"] = d.get("eval_loss")
            return
        if d.get("loss") is None:
            return  # 训练结束汇总字典（train_runtime 等）不计入
        state["loss"] = d["loss"]
        if state["loss"] is not None and state["first_loss"] is None:
            state["first_loss"] = state["loss"]
        if state["loss"] is not None:
            state["min_loss"] = (state["min_loss"] if state["min_loss"] is not None
                                 else state["loss"])
            state["min_loss"] = min(state["min_loss"], state["loss"])
        state["epoch"] = d.get("epoch", state["epoch"])
        state["lr"] = d.get("learning_rate", state["lr"])
        state["count"] += 1
        state["step"] = int(d.get("step") or state["count"] * 10)
        loss_s = f"loss={state['loss']:.4f}" if state["loss"] is not None else "loss=n/a"
        ev_s = f" eval={state['eval_loss']:.4f}" if state["eval_loss"] is not None else ""
        lr_s = f" lr={state['lr']:.2e}" if state["lr"] is not None else ""
        print(f"  ==> epoch {state['epoch']:.2f} | step {state['step']}/{total} | {loss_s}{ev_s}{lr_s}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=base_env(), cwd=str(run_dir))
    rc = stream_process(proc, console, handler)
    if rc != 0:
        raise RuntimeError(f"明文训练失败 rc={rc}，见 {console}")

    print("[明文] 训练完成，开始测试集评估 ...")
    eval_cmd = [
        str(PY), "-u", "-s",
        str(ROOT / "coordinator" / "evaluate_auprc.py"),
        "--model_id", SNAP,
        "--adapter", str(run_dir),
        "--data", str(data_dir / "test.jsonl"),
        "--out", str(run_dir / "test_metrics.json"),
        "--task_type", dataset,
        "--class_outputs", json.dumps(class_outputs),
        "--max_seq_length", str(max_seq_len),
        "--train_data", str(data_dir / "train.jsonl"),
    ]
    prior_c = float(_env("CF_PRIOR_CORRECTION") or 0)
    if prior_c > 0:
        eval_cmd += ["--prior_correction", str(prior_c)]
    ev = subprocess.run(eval_cmd, env=base_env(), cwd=str(run_dir),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(ev.stdout.decode("utf-8", "replace")[-2000:])
    if ev.returncode != 0:
        raise RuntimeError(f"明文评估失败 rc={ev.returncode}")

    test = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    summary = {
        "mode": "plaintext-baseline",
        "label": MODES[1]["label"],
        "run_dir": str(run_dir),
        "dataset": dataset,
        "n_train": n_train,
        "params": params,
        "steps": state["step"],
        "train": {
            "first_loss": state.get("first_loss"),
            "min_loss": state.get("min_loss"),
            "final_loss": state.get("loss"),
            "best_eval_loss": state.get("eval_loss"),
            "epochs_done": state.get("epoch"),
        },
        "test": test,
    }
    return summary


# --------------------------------------------------------------------------- #
#  三进程模式
# --------------------------------------------------------------------------- #
STEP3_RE = re.compile(r"(?:Step (\d+)|\[max_train_steps\] step=(\d+)).*?loss=([\d.eE+-]+)")
EPOCH3_RE = re.compile(
    r"Epoch (\d+): train_loss(?:_proxy)?=([\d.]+) \| val_ce_loss=([\d.]+)"
    r"(?: \| val_AUPRC=([\d.]+) \| val_AUC=([\d.]+) \| val_acc@0\.5=([\d.]+)"
    r"| \| val_micro_F1=([\d.]+) \(P=([\d.]+) R=([\d.]+) Acc=([\d.]+)\)"
    r" \| val_macro_F1=([\d.]+) \| val_weighted_F1=([\d.]+))"
)


def run_three(run_dir: Path, params: dict, max_steps: int, mode: dict, dry_run: bool,
              dataset: str = DEFAULT_TASK) -> dict:
    print(f"\n[{mode['label']}] 输出目录: {run_dir}")
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # 运行级配置：产物全部落在 run 目录，加密相关指向 /root/autodl-tmp 缓存
    src_cfg = ROOT / "coordinator" / mode["config"]
    cfg = json.loads(src_cfg.read_text(encoding="utf-8"))
    # ---- Task/dataset overrides (selected before running) ----
    profile = get_profile(dataset)
    cfg["task_type"] = dataset
    cfg["data_dir"] = profile.get("data_dir", cfg.get("data_dir"))
    cfg["max_seq_length"] = int(profile.get("max_seq_length", cfg.get("max_seq_length", 128)))
    cfg["dp_num_classes"] = int(profile.get("dp_num_classes", cfg.get("dp_num_classes", 2)))
    cfg["class_outputs"] = list(profile.get("class_outputs", cfg.get("class_outputs", ["Yes", "No"])))
    cfg["eval_mode"] = profile.get("eval_mode", cfg.get("eval_mode", "binary"))
    if _env("CF_DP_ENABLE"):
        cfg["dp_enable"] = _env("CF_DP_ENABLE") == "1"
        print(f"  [env] CF_DP_ENABLE -> dp_enable={cfg['dp_enable']}")
    n_train = _count_jsonl_rows(str(cfg["data_dir"]))
    cfg["eval_script"] = str(ROOT / "coordinator" / "evaluate_auprc.py")
    cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    cfg["log_dir"] = str(logs)
    cfg["adapter_dir"] = str(run_dir / "adapter")
    cfg["batch_size"] = params["batch_size"]
    cfg["max_epochs"] = params["epochs"]
    crypto = prepare_crypto(mode, run_dir.name)
    cfg["hints_dir"] = str(crypto["enc_db"] / "s3pir_hints")
    cfg["rms_hints_dir"] = str(crypto["rms_hints"])
    cfg["rms_db_dir"] = str(crypto["rms_db"])
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        print("[DRY-RUN] 将启动 U/M/S 三节点 + 平台(:8600) 并运行 coordinator，现停止于启动前")
        return {"dry_run": True, "run_dir": str(run_dir), "config": str(run_dir / "config.json")}

    print("[三进程] 清理残留节点/平台 ...")
    kill_stale_nodes()

    env = base_env()
    u_log = (logs / "party_u.log").open("a", encoding="utf-8")
    m_log = (logs / "party_m.log").open("a", encoding="utf-8")
    s_log = (logs / "party_s.log").open("a", encoding="utf-8")
    presets_json = str(ROOT / "data" / "fixtures" / "eval-presets.json")
    stable_ck = ROOT / "party_m" / "checkpoints" / "best_checkpoint.pt"
    fallback_ck = str(stable_ck if stable_ck.exists()
                      else run_dir / "checkpoints" / "best_checkpoint.pt")
    print("[三进程] 启动 U/M/S 节点：M 生成 BFV 密钥，S 构建密文库与 hints，"
          "U 提供握手/兜底/评测（进度实时显示在下方）...")
    u_proc = m_proc = s_proc = None
    u_proc = subprocess.Popen(
        [str(PY), "-u", "-s", str(ROOT / "party_u" / "main_u.py"),
         "--port", "9001", "--data_dir", str(ROOT / "party_u" / "data"),
         "--metrics_dir", str(logs), "--presets_json", presets_json],
        stdout=u_log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT), start_new_session=True,
    )
    m_proc = subprocess.Popen(
        [str(PY), "-u", "-s", str(ROOT / "party_m" / "main_m.py"),
         "--port", "9002", "--keys_dir", str(crypto["m_keys"]), "--model_path", SNAP,
         "--metrics_dir", str(logs), "--presets_json", presets_json,
         "--fallback_checkpoint", fallback_ck],
        stdout=m_log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT), start_new_session=True,
    )
    s_proc = subprocess.Popen(
        [str(PY), "-u", "-s", str(ROOT / "party_s" / "main_s.py"),
         "--port", "9003", "--db_dir", str(crypto["enc_db"]), "--model_path", SNAP,
         "--metrics_dir", str(logs), "--presets_json", presets_json],
        stdout=s_log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT), start_new_session=True,
    )

    stop_tail = threading.Event()

    def _tail(name: str, path: Path) -> None:
        last = 0
        while not stop_tail.is_set():
            try:
                size = path.stat().st_size
                if size > last:
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(last)
                        for ln in f:
                            print(f"  [{name}] {ln.rstrip()}")
                        last = f.tell()
            except Exception:
                pass
            time.sleep(0.5)

    t_u = threading.Thread(target=_tail, args=("U", logs / "party_u.log"), daemon=True)
    t_m = threading.Thread(target=_tail, args=("M", logs / "party_m.log"), daemon=True)
    t_s = threading.Thread(target=_tail, args=("S", logs / "party_s.log"), daemon=True)
    t_u.start()
    t_m.start()
    t_s.start()
    plat_proc = None
    plat_log = None
    completed = False
    try:
        print("[三进程] 等待 U/M/S 节点就绪 ...")
        if not (wait_hello(9001) and wait_hello(9002) and wait_hello(9003)):
            raise RuntimeError("U/M/S 节点启动失败，见 logs/party_*.log")
        print("[三进程] U/M/S 就绪")

        # 演枢台平台（docs/00：总控台 :8600）
        demo_logs = ROOT / "demo" / "logs"
        demo_logs.mkdir(parents=True, exist_ok=True)
        plat_log_path = demo_logs / f"platform-{run_dir.name}.log"
        plat_log = plat_log_path.open("a", encoding="utf-8")
        plat_proc = subprocess.Popen(
            ["node", "server/index.js"],
            cwd=str(ROOT / "platform"), env=env, start_new_session=True,
            stdout=plat_log, stderr=subprocess.STDOUT,
        )
        if not wait_url("http://127.0.0.1:8600/api/state", timeout=60):
            raise RuntimeError(f"平台启动失败，见 {plat_log_path}")
        print(f"[三进程] 平台就绪：http://127.0.0.1:8600（日志 {plat_log_path}）")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8600/api/nodes/probe", timeout=30) as r:
                st = json.loads(r.read().decode("utf-8"))
            print("[平台] 探活结果: " + json.dumps(st["nodes"], ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print(f"[平台] 探活失败（不影响训练）: {exc}")

        steps_per_epoch = max(1, math.ceil(n_train / params["batch_size"])) if n_train else max(1, 10000 // params["batch_size"])
        total = max_steps if max_steps else int(steps_per_epoch * params["epochs"])
        state = {"step": 0, "loss": None, "epoch_metrics": [], "last_val_ce": None}

        def _lr_at(st: int) -> float:
            """与 party_m._build_lr_scheduler 相同的 warmup+cosine 调度。"""
            s = max(0, st - 1)
            warmup = int(params.get("warmup_steps", 100))
            peak = float(params.get("learning_rate", 2e-4))
            if warmup and s < warmup:
                lam = (s + 1) / warmup
            else:
                t = (s - warmup) / max(1, total - warmup)
                lam = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
            return peak * lam

        def handler(line: str) -> None:
            m3 = EPOCH3_RE.search(line)
            if m3:
                e = int(m3.group(1))
                row = {
                    "epoch": e,
                    "train_loss": float(m3.group(2)),
                    "val_ce_loss": float(m3.group(3)),
                }
                if m3.group(4) is not None:   # ClinVar binary epoch line
                    row.update({
                        "val_auprc": float(m3.group(4)),
                        "val_auc": float(m3.group(5)),
                        "val_acc": float(m3.group(6)),
                    })
                else:                         # BioTriplex multiclass epoch line
                    row.update({
                        "val_micro_f1": float(m3.group(7)),
                        "val_micro_p": float(m3.group(8)),
                        "val_micro_r": float(m3.group(9)),
                        "val_acc": float(m3.group(10)),
                        "val_macro_f1": float(m3.group(11)),
                        "val_weighted_f1": float(m3.group(12)),
                    })
                state["epoch_metrics"].append(row)
                state["last_val_ce"] = row["val_ce_loss"]
                auprc_s = f" | val_AUPRC={row['val_auprc']:.4f}" if row.get("val_auprc") is not None else ""
                micro_s = f" | val_micro_F1={row['val_micro_f1']:.4f}" if row.get("val_micro_f1") is not None else ""
                print(f"  [epoch {e}] train_loss={row['train_loss']:.4f} | "
                      f"val_ce_loss={row['val_ce_loss']:.4f}{auprc_s}{micro_s} | "
                      f"val_acc={row['val_acc']:.4f}")
                return
            m1 = STEP3_RE.search(line)
            if m1:
                st = int(m1.group(1) or m1.group(2))
                loss = float(m1.group(3))
                state["step"] = st
                state["loss"] = loss
                ep = st / steps_per_epoch
                ev_s = (f" eval={state['last_val_ce']:.4f}"
                        if state["last_val_ce"] is not None else "")
                print(f"  ==> epoch {ep:.2f} | step {st}/{total} | loss={loss:.4f}"
                      f"{ev_s} lr={_lr_at(st):.2e}")

        coord_cmd = [
            str(PY), "-u", "-s", str(ROOT / "coordinator" / "main.py"),
            "--config", str(run_dir / "config.json"),
            "--batch_size", str(params["batch_size"]),
            "--log_freq", "10",
        ]
        if max_steps:
            coord_cmd += ["--max_train_steps", str(max_steps), "--max_epochs", "1"]
        else:
            coord_cmd += ["--max_epochs", str(params["epochs"])]
        if _env("CF_SKIP_TRAIN") == "1":
            # L1 兜底：跳过训练，加载预微调检查点后直接评测（docs/04）
            run_ck_dir = run_dir / "checkpoints"
            run_ck_dir.mkdir(parents=True, exist_ok=True)
            stable_last = ROOT / "party_m" / "checkpoints" / "last_checkpoint.pt"
            if not (run_ck_dir / "last_checkpoint.pt").exists() and stable_last.exists():
                shutil.copy(stable_last, run_ck_dir / "last_checkpoint.pt")
                print(f"[三进程] 复制预微调检查点 → {run_ck_dir / 'last_checkpoint.pt'}")
            coord_cmd += ["--skip_train"]
        print("[三进程] 启动 coordinator：" + " ".join(coord_cmd))
        coord_log = logs / "coordinator.log"
        coord_proc = subprocess.Popen(coord_cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        rc = stream_process(coord_proc, coord_log, handler)
        if rc != 0:
            raise RuntimeError(f"coordinator 失败 rc={rc}，见 {coord_log}")
        completed = True
    finally:
        stop_tail.set()
        time.sleep(0.6)
        if not completed or _env("CF_KEEP_SERVICES", "1") == "0":
            for p in (u_proc, m_proc, s_proc):
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
            if plat_proc is not None:
                try:
                    os.killpg(os.getpgid(plat_proc.pid), signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(1)
            for p in (u_proc, m_proc, s_proc):
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
            if plat_proc is not None:
                try:
                    os.killpg(os.getpgid(plat_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        u_log.close()
        m_log.close()
        s_log.close()
        if plat_log is not None:
            try:
                plat_log.close()
            except Exception:
                pass

    # L1 兜底检查点：仓库稳定路径不存在时，把本次训练 best 复制过去
    if not stable_ck.exists() and (run_dir / "checkpoints" / "best_checkpoint.pt").exists():
        stable_ck.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(run_dir / "checkpoints" / "best_checkpoint.pt", stable_ck)
        print(f"[三进程] 已复制 best_checkpoint → {stable_ck}（L1 兜底检查点）")
    if completed and _env("CF_KEEP_SERVICES", "1") != "0":
        print("\n[三进程] U/M/S 节点与平台保持运行：")
        print("  U :9001 | M :9002 | S :9003 | 平台 :8600")
        print("  平台页面: http://127.0.0.1:8600/")
        print("  联调接口: POST /api/nodes/probe、/api/fallback/pretrained、/api/eval/run")
        print("  设 CF_KEEP_SERVICES=0 可让脚本结束后自动清理。")

    # 收集产物与测试指标
    tms = sorted(logs.glob("training_metrics_*.json"))
    epochs = []
    if tms:
        epochs = json.loads(tms[-1].read_text(encoding="utf-8"))
    auprc_path = logs / f"{dataset}_eval.json"
    if not auprc_path.exists():
        auprc_path = logs / "clinvar_auprc.json"
    test = json.loads(auprc_path.read_text(encoding="utf-8")) if auprc_path.exists() else {}
    if auprc_path.exists():
        shutil.copy(auprc_path, run_dir / "test_metrics.json")

    summary = {
        "mode": mode["tag"],
        "label": mode["label"],
        "run_dir": str(run_dir),
        "dataset": dataset,
        "n_train": n_train,
        "params": params,
        "steps": state["step"],
        "epochs_table": [
            {
                "epoch": r.get("epoch"),
                "train_loss": r.get("train_loss"),
                "val_ce_loss": r.get("val_ce_loss"),
                "val_auprc": r.get("val_auprc"),
                "val_auc": r.get("val_auc"),
                "val_acc": r.get("val_accuracy"),
            }
            for r in epochs
        ] or state["epoch_metrics"],
        "test": test,
    }
    return summary


# --------------------------------------------------------------------------- #
#  绘图与汇总
# --------------------------------------------------------------------------- #
def plot_plain(run_dir: Path, summary: dict) -> str:
    if not HAS_MPL:
        return ""
    steps, losses, ev_steps, ev_losses = [], [], [], []
    train_log = run_dir / "train.log"
    if train_log.exists():
        for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
            d = _hf_dict_from_line(line)
            if d is None:
                continue
            if d.get("loss") is not None:
                steps.append(int(d.get("step") or len(steps) + 1))
                losses.append(float(d["loss"]))
            if d.get("eval_loss") is not None:
                ev_steps.append(int(d.get("step") or len(ev_steps) + 1))
                ev_losses.append(float(d["eval_loss"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    if steps:
        ax.plot(steps, losses, label="train loss")
    if ev_steps:
        ax.plot(ev_steps, ev_losses, "o--", label="eval loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Plaintext LoRA Baseline — loss convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    out = run_dir / "loss_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def plot_three(run_dir: Path, summary: dict) -> str:
    if not HAS_MPL:
        return ""
    steps, losses = [], []
    coord_log = run_dir / "logs" / "coordinator.log"
    if coord_log.exists():
        for line in coord_log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = STEP3_RE.search(line)
            if m:
                steps.append(int(m.group(1) or m.group(2)))
                losses.append(float(m.group(3)))
    epochs = summary.get("epochs_table") or []
    fig, ax = plt.subplots(figsize=(9, 5))
    if steps:
        ax.plot(steps, losses, label="train loss_proxy")
    ax.set_xlabel("step")
    ax.set_ylabel("loss_proxy", color="tab:blue")
    ax.set_title(f"{summary['mode']} — loss convergence")
    ax.grid(alpha=0.3)
    if epochs:
        ax2 = ax.twinx()
        es = [r["epoch"] for r in epochs if r.get("val_ce_loss") is not None]
        vc = [r["val_ce_loss"] for r in epochs if r.get("val_ce_loss") is not None]
        if es:
            ax2.plot(es, vc, "ro--", label="val_ce_loss")
        ax2.set_ylabel("val_ce_loss", color="tab:red")
    lines1, lab1 = ax.get_legend_handles_labels()
    ax2 = locals().get("ax2")
    lines2, lab2 = (ax2.get_legend_handles_labels() if ax2 is not None else ([], []))
    ax.legend(lines1 + lines2, lab1 + lab2)
    out = run_dir / "loss_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def print_summary(mode: dict, summary: dict) -> None:
    print("\n" + "=" * 78)
    print(f"运行完成：{mode['label']}")
    print(f"运行目录：{summary['run_dir']}")
    print(f"总步数：{summary.get('steps', 'n/a')}")
    test = summary.get("test") or {}
    print("— 训练过程 —")
    if summary.get("epochs_table"):
        def _fmt(v):
            try:
                return f"{float(v):.4f}"
            except (TypeError, ValueError):
                return "n/a"
        print(f"  {'epoch':<7}{'train_loss':<12}{'val_ce_loss':<12}{'val_AUPRC':<10}{'val_AUC':<10}{'val_acc':<10}")
        for r in summary["epochs_table"]:
            print(f"  {str(r.get('epoch','?')):<7}{_fmt(r.get('train_loss')):<12}"
                  f"{_fmt(r.get('val_ce_loss')):<12}{_fmt(r.get('val_auprc')):<10}"
                  f"{_fmt(r.get('val_auc')):<10}{_fmt(r.get('val_acc')):<10}")
    elif summary.get("train"):
        t = summary["train"]
        print(f"  first_loss={t.get('first_loss')}  min_loss={t.get('min_loss')}  "
              f"final_loss={t.get('final_loss')}  best_eval_loss={t.get('best_eval_loss')}")
    print("— 测试集（test.jsonl）—")
    if test:
        if test.get("task_type") == "multiclass":
            print(f"  n={test.get('n')}  classes={test.get('n_classes')}")
            print(f"  accuracy={test.get('accuracy')}  macro_f1={test.get('macro_f1')}  "
                  f"weighted_f1={test.get('weighted_f1')}")
        else:
            print(f"  n={test.get('n')}  pos_rate={test.get('pos_rate')}")
            print(f"  AUPRC={test.get('auprc')}  AUC={test.get('auc')}  "
                  f"acc@0.5={test.get('accuracy@0.5')}  macro_f1={test.get('macro_f1')}")
            pg = test.get("per_gene") or {}
            print(f"  per_gene: mean_auprc={pg.get('mean_auprc')} (n_genes={pg.get('n_genes')})")
    png = summary.get("loss_curve_png")
    if png:
        print(f"损失收敛图：{png}")
    print("=" * 78)


def plot_comparison(summaries: list, out_png: Path) -> str:
    if not HAS_MPL or len(summaries) < 2:
        return ""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1 = axes[0]
    for s in summaries:
        mode = s["mode"]
        run_dir = Path(s["run_dir"])
        batch = (s.get("params") or {}).get("batch_size", 16)
        n_train = s.get("n_train") or 0
        steps_per_epoch = max(1, math.ceil(n_train / batch)) if n_train else max(1, 10000 // batch)
        xs, ys = [], []
        if mode == "plaintext-baseline":
            tr = run_dir / "train.log"
            if tr.exists():
                for line in tr.read_text(encoding="utf-8", errors="replace").splitlines():
                    d = _hf_dict_from_line(line)
                    if d is None:
                        continue
                    if d.get("loss") is not None:
                        xs.append((d.get("step", 0) or 0) / steps_per_epoch)
                        ys.append(float(d["loss"]))
        else:
            cl = run_dir / "logs" / "coordinator.log"
            if cl.exists():
                for line in cl.read_text(encoding="utf-8", errors="replace").splitlines():
                    m = STEP3_RE.search(line)
                    if m:
                        st = int(m.group(1) or m.group(2))
                        xs.append(st / steps_per_epoch)
                        ys.append(float(m.group(3)))
        if xs:
            ax1.plot(xs, ys, label=mode, linewidth=1.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss convergence (train)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    labels = [s["mode"] for s in summaries]
    auprcs = []
    for s in summaries:
        t = s.get("test") or {}
        auprcs.append(float(t.get("auprc") if t.get("auprc") is not None else t.get("accuracy", 0.0)))
    ax2.bar(labels, auprcs, color=["#4C72B0", "#DD8452", "#55A868"])
    ax2.set_ylabel("test AUPRC/accuracy")
    ax2.set_title("Test AUPRC by mode")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(auprcs):
        ax2.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    ax3 = axes[2]
    f1s = [(s.get("test") or {}).get("macro_f1", 0.0) for s in summaries]
    ax3.bar(labels, f1s, color=["#4C72B0", "#DD8452", "#55A868"])
    ax3.set_ylabel("test macro_f1")
    ax3.set_title("Test Macro-F1 by mode")
    ax3.set_ylim(0, 1)
    for i, v in enumerate(f1s):
        ax3.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return str(out_png)


def print_comparison(summaries: list) -> None:
    def _fmt(v):
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "n/a"
    print("\n" + "=" * 78)
    print("三模式对比汇总")
    print(f"  {'模式':<22}{'test AUPRC':<12}{'test AUC':<12}{'acc@0.5':<10}"
          f"{'macro_f1':<10}{'最佳val指标':<14}")
    for s in summaries:
        t = s.get("test") or {}
        best_val = None
        for r in s.get("epochs_table") or []:
            v = r.get("val_auprc") if r.get("val_auprc") is not None else r.get("val_acc")
            if v is not None:
                best_val = max(best_val or -1, float(v))
        print(f"  {s['label']:<22}{_fmt(t.get('auprc')):<12}{_fmt(t.get('auc')):<12}"
              f"{_fmt(t.get('accuracy@0.5')):<10}{_fmt(t.get('macro_f1')):<10}{_fmt(best_val):<14}")
    print("=" * 78)


def save_summary(run_dir: Path, summary: dict) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args()

    dataset = choose_dataset()
    banner(dataset)
    disk_warning()
    modes = choose_modes()
    params = edit_params()
    out_root = choose_out_root()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summaries = []
    for mid in modes:
        mode = MODES[mid]
        show_mode_params(mode, dataset)
        run_dir = run_dir_for(out_root, mode)
        print(f"已创建运行目录：{run_dir}")
        interactive = sys.stdin.isatty()
        if interactive and not _env("CF_MAX_STEPS") and not args.dry_run:
            input("按 Enter 启动微调（Ctrl+C 可取消）...")
        max_steps = int(_env("CF_MAX_STEPS")) if _env("CF_MAX_STEPS") else 0
        if mode["kind"] == "plain":
            summary = run_plain(run_dir, params, max_steps, args.dry_run, dataset)
        else:
            summary = run_three(run_dir, params, max_steps, mode, args.dry_run, dataset)
        if args.dry_run:
            print("[DRY-RUN] 结束，未启动任何训练")
            continue
        summary["loss_curve_png"] = (
            plot_plain(run_dir, summary) if mode["kind"] == "plain"
            else plot_three(run_dir, summary)
        )
        save_summary(run_dir, summary)
        summaries.append(summary)
        print_summary(mode, summary)

    if summaries and len(summaries) > 1:
        comp_png = Path(out_root) / f"comparison_{ts}.png"
        plot_comparison(summaries, comp_png)
        print_comparison(summaries)
        print(f"对比图：{comp_png}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[错误] {exc}")
        sys.exit(1)
