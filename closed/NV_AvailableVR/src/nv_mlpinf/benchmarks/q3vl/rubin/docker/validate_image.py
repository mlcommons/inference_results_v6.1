#!/usr/bin/env python3
"""Validation of the Q3VL Rubin image (runs inside the image with /usr/bin/python3).

  --build     CPU only, at image build time (no driver): versions of the stack, the FlashInfer
              artifact pin and the prefetched cute-dsl FMHA leaf, the NIXL EP build for the
              installed torch, `uv pip check` (recorded), and the vLLM / Dynamo import checks.
  --gpu       with a driver (a GPU smoke test on the build host or on a Rubin node): per check a
              fresh process with exit code 0 and 0 Tracebacks:
                g1 torch: version, CUDA, arch list, device cc, a bf16 matmul checked against fp32 CPU;
                g2 NCCL: init + all_reduce over every visible GPU (one process per GPU);
                g3 cute-dsl FMHA: every export of the pinned leaf (or --names-file) loaded through
                   FlashInfer's pinned-artifact path (_load_from_artifact), FLASHINFER_NO_DOWNLOAD=1,
                   no FLASHINFER_DSL_FMHA_LOCAL_DIR; each .so mapped from FLASHINFER_CUBIN_DIR;
                a  torch + nixl_ep -> nixl_ep_cpp_torch<major><minor>;
                b/c NIXL UCX loopback DRAM and VRAM (GPU0 <- GPU1, or 0 <- 0 with one GPU);
                d  vLLM check_nixl_version + NixlConnector;
                e1-e3 dynamo.vllm.main, dynamo.frontend, the qwen3_vl_moe worker import (has_nixl_ep);
                m  /proc/self/maps after cuBLAS, cuDNN, NCCL and a NIXL UCX agent are in use.
Every libuc* must come from the NIXL wheel (nixl_cu13*), with at most one libucp.
Prints VALIDATE_IMAGE_PASS / _FAIL and writes --report JSON.
"""

import argparse
import importlib.metadata as md
import json
import os
import subprocess
import sys
import tempfile

TRACEBACK = "Traceback (most recent call last)"
CHILD_TIMEOUT_S = 900
SITE = "/usr/local/lib/python3.12/dist-packages"

PRELUDE = r'''
import faulthandler, os, re, sys
faulthandler.dump_traceback_later(%(watchdog)d, exit=True)

def _maps():
    out = []
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.split(None, 5)
            if len(p) == 6 and p[5].strip().startswith("/"):
                out.append(p[5].strip())
    return sorted(set(out))

def _ucx_guard(tag):
    ucx = [p for p in _maps() if re.match(r"libuc[mpst]([-_.]|$)", os.path.basename(p))]
    bad = [p for p in ucx if "/nixl_cu13" not in p]
    assert not bad, "UCX mapped from outside the NIXL wheel: %%s" %% bad
    libucp = [p for p in ucx if os.path.basename(p).startswith("libucp")]
    assert len(libucp) <= 1, "more than one libucp mapped: %%s" %% libucp
    print("UCX_GUARD_OK", tag, "ucx_objects=%%d" %% len(ucx), "libucp=%%s" %%
          (libucp[0] if libucp else "not loaded"), flush=True)
'''

G1_TORCH = r'''
import torch
print("TORCH", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "nccl", torch.cuda.nccl.version(), flush=True)
print("ARCH_LIST", " ".join(torch.cuda.get_arch_list()), flush=True)
n = torch.cuda.device_count()
assert n >= 1, "no GPU"
for d in range(n):
    p = torch.cuda.get_device_properties(d)
    print("DEVICE", d, p.name, "cc %%d.%%d" %% (p.major, p.minor), "sms", p.multi_processor_count, flush=True)
g = torch.Generator().manual_seed(0)
a = torch.randn(512, 768, generator=g)
b = torch.randn(768, 384, generator=g)
ref = a @ b
for d in range(n):
    out = (a.to("cuda:%%d" %% d, torch.bfloat16) @ b.to("cuda:%%d" %% d, torch.bfloat16)).float().cpu()
    err = ((out - ref).norm() / ref.norm()).item()
    assert err < 2e-2, (d, err)
    print("MATMUL_OK dev", d, "bf16 rel_err %%.2e" %% err, flush=True)
x = torch.randn(4, 16, 32, 32, generator=g)
w = torch.randn(8, 16, 3, 3, generator=g)
yr = torch.nn.functional.conv2d(x, w)
errs = {}
for tf32 in (True, False):  # cuDNN convs default to TF32 (allow_tf32=True); check both modes
    torch.backends.cudnn.allow_tf32 = tf32
    y = torch.nn.functional.conv2d(x.cuda(), w.cuda()).cpu()
    errs[tf32] = ((y - yr).norm() / yr.norm()).item()
torch.backends.cudnn.allow_tf32 = True
assert errs[False] < 1e-5 and errs[True] < 5e-3, errs
print("CONV_OK cudnn", torch.backends.cudnn.version(), "rel_err fp32 %%.1e tf32 %%.1e" %% (errs[False], errs[True]), flush=True)
'''

G2_NCCL = r'''
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def _worker(rank, world, port):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:%%d" %% port, rank=rank, world_size=world)
    t = torch.full((1 << 20,), float(rank + 1), device="cuda")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    want = world * (world + 1) / 2
    assert float(t[0]) == want and float(t[-1]) == want, (float(t[0]), want)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world = torch.cuda.device_count()
    assert world >= 1
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(_worker, args=(world, port), nprocs=world, join=True)
    print("NCCL_OK world", world, "nccl", torch.cuda.nccl.version(), flush=True)
'''

G3_FMHA = r'''
import hashlib
os.environ.pop("FLASHINFER_DSL_FMHA_LOCAL_DIR", None)
os.environ["FLASHINFER_NO_DOWNLOAD"] = "1"
import torch
import flashinfer
from flashinfer.attention.cute_dsl import fmha
from flashinfer.jit.env import FLASHINFER_CUBIN_DIR
sm = %(sm)r
names = %(names)r
leaf = os.path.join(str(FLASHINFER_CUBIN_DIR), fmha.DSL_FMHA_ARTIFACT_PATH, fmha._get_cpu_arch(), sm)
checks = fmha._get_checksums(sm)
if not names:
    names = sorted(checks)
print("FMHA_PIN", fmha.DSL_FMHA_ARTIFACT_PATH, "cubin_dir", FLASHINFER_CUBIN_DIR, "names", len(names), flush=True)
missing = [n for n in names if n not in checks]
assert not missing, "not in the pinned manifest: %%s" %% missing[:5]
for n in names:
    fn = fmha._load_from_artifact(n, sm, enable_tvm_ffi=n.endswith("_tvmffi"))
    assert fn is not None, n
mapped = [p for p in _maps() if p.startswith(leaf + "/") and p.endswith(".so")]
assert len(mapped) == len(names), (len(mapped), len(names))
print("FMHA_LOADED %%d/%%d via the pinned-artifact path, mapped from %%s" %% (len(names), len(names), leaf), flush=True)
'''

CHECK_A = r'''
import torch, nixl_ep
cpp = sys.modules.get("nixl_ep_cu13.nixl_ep_cpp")
assert cpp is not None, "nixl_ep did not dispatch to nixl_ep_cu13"
torch_mm = "".join(torch.__version__.split(".")[:2])
assert os.path.basename(cpp.__file__).startswith("nixl_ep_cpp_torch" + torch_mm), cpp.__file__
print("NIXL_EP_OK", cpp.__file__, flush=True)
_ucx_guard("nixl_ep")
'''

LOOPBACK = r'''
import time
import numpy as np
import torch
try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:  # no `nixl` meta package: use the CUDA-13 package directly
    from nixl_cu13._api import nixl_agent, nixl_agent_config

MEM = %(mem)r
DEV, PEER = %(dev)d, %(peer)d
place = (lambda d: "cuda:%%d" %% d) if MEM == "VRAM" else (lambda d: "cpu")
did = (lambda d: d) if MEM == "VRAM" else (lambda d: 0)
blk, stride, n = 4096, 8192, 64
src = (torch.arange(n * stride, device=place(PEER)) %% 251).to(torch.uint8)
dst = torch.zeros(n * stride, dtype=torch.uint8, device=place(DEV))
if MEM == "VRAM":
    torch.cuda.synchronize()
A = nixl_agent("gateA", nixl_agent_config(backends=["UCX"]))
B = nixl_agent("gateB", nixl_agent_config(backends=["UCX"]))
print("plugins", A.get_plugin_list(), "UCX mem types", A.get_backend_mem_types("UCX"), flush=True)
A.register_memory([dst])
B.register_memory([src])
remote = A.add_remote_agent(B.get_agent_metadata())
rows = lambda t, d: np.array([[t.data_ptr(), blk, d, stride, n]], dtype=np.uint64)
loc = A.prep_xfer_dlist("NIXL_INIT_AGENT", rows(dst, did(DEV)), mem_type=MEM)
rem = A.prep_xfer_dlist(remote, rows(src, did(PEER)), mem_type=MEM)
h = A.make_prepped_xfer("READ", loc, list(range(n)), rem, list(range(n)), b"gate")
t0 = time.time()
state = A.transfer(h)
assert state != "ERR", state
state = A.check_xfer_state(h)
while state == "PROC":
    assert time.time() - t0 < 120, "transfer timed out"
    state = A.check_xfer_state(h)
assert state == "DONE", state
print("backend", A.query_xfer_backend(h), flush=True)
if MEM == "VRAM":
    torch.cuda.synchronize()
mask = (torch.arange(n * stride) %% stride) < blk
d, s = dst.cpu(), src.cpu()
assert torch.equal(d[mask], s[mask]), "payload mismatch"
assert int(d[~mask].sum()) == 0, "bytes outside the blocks were written"
A.release_xfer_handle(h)
print("LOOPBACK_OK", MEM, "Nx5", "dev", DEV, "<-", PEER, flush=True)
_ucx_guard("loopback-" + MEM)
'''

CHECK_D = r'''
from vllm.distributed import nixl_utils
from vllm.distributed.nixl_utils import check_nixl_version
check_nixl_version(%(nixl)r)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
cls = KVConnectorFactory.get_connector_class_by_name("NixlConnector")
wrapper = nixl_utils.NixlWrapper
assert wrapper is not None and wrapper.__module__.startswith("nixl_cu13."), wrapper
print("CHECK_NIXL_VERSION_OK", %(nixl)r, "NixlConnector ->", cls.__module__ + "." + cls.__name__,
      "NixlWrapper ->", wrapper.__module__ + "." + wrapper.__name__, flush=True)
_ucx_guard("vllm-nixl")
'''

CHECK_E = {
    "e1_dynamo_vllm_main": r'''
import dynamo.vllm.main
print("IMPORT_OK dynamo.vllm.main", dynamo.vllm.main.__file__, flush=True)
_ucx_guard("dynamo.vllm.main")
''',
    "e2_dynamo_frontend": r'''
import dynamo.frontend
print("IMPORT_OK dynamo.frontend", dynamo.frontend.__file__, flush=True)
_ucx_guard("dynamo.frontend")
''',
    "e3_worker_model_import": r'''
import vllm.model_executor.models.qwen3_vl_moe
from vllm.utils.import_utils import has_nixl_ep
print("HAS_NIXL_EP", has_nixl_ep(), "nixl_ep_loaded", "nixl_ep" in sys.modules, flush=True)
if %(native)r:
    assert has_nixl_ep(), "nixl_ep not importable with the driver present"
_ucx_guard("qwen3_vl_moe")
''',
}

MAPS = r'''
import torch
import torch.distributed as dist
import numpy as np
try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:  # no `nixl` meta package: use the CUDA-13 package directly
    from nixl_cu13._api import nixl_agent, nixl_agent_config
a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(a @ a).sum().item()
torch.nn.functional.conv2d(torch.randn(1, 4, 8, 8, device="cuda"), torch.randn(4, 4, 3, 3, device="cuda")).sum().item()
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
dist.init_process_group("nccl", init_method="tcp://127.0.0.1:%%d" %% port, rank=0, world_size=1)
t = torch.ones(16, device="cuda"); dist.all_reduce(t); torch.cuda.synchronize()
agent = nixl_agent("mapsA", nixl_agent_config(backends=["UCX"]))
dist.destroy_process_group()
keys = ("libcublas", "libcublasLt", "libcudnn", "libnccl", "libucp", "libuct", "libucs", "libcudart",
        "libnvrtc", "libnvJitLink", "libcuda.so", "libtorch_cuda", "libcusparseLt", "libnvshmem")
for p in _maps():
    b = os.path.basename(p)
    if any(b.startswith(k) for k in keys):
        print("MAPPED", b, p, flush=True)
_ucx_guard("maps")
print("MAPS_OK", flush=True)
'''


def child_env(extra=None):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("UCX_") or key in ("PYTHONPATH", "LD_PRELOAD", "NIXL_UCX_DEEPBIND",
                                              "FLASHINFER_DSL_FMHA_LOCAL_DIR"):
            env.pop(key)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


def run_child(name, code, results, enforce_no_traceback=True, extra_env=None, keep=()):
    prelude = PRELUDE % {"watchdog": CHILD_TIMEOUT_S - 30}
    # A real script file (not -c): torch.multiprocessing spawn re-imports __main__.
    fd, script = tempfile.mkstemp(prefix=f"q3vl_rubin_{name}_", suffix=".py", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write(prelude + code)
    try:
        p = subprocess.run(["/usr/bin/python3", "-X", "faulthandler", script], cwd="/tmp",
                           env=child_env(extra_env), capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        rc = "timeout"
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    finally:
        os.unlink(script)
    tb = out.count(TRACEBACK) + err.count(TRACEBACK)
    ok = rc == 0 and (tb == 0 or not enforce_no_traceback)
    sig = f" (signal {-rc})" if isinstance(rc, int) and rc < 0 else ""
    print(f"CHECK {name}: rc={rc}{sig} tracebacks={tb} -> {'PASS' if ok else 'FAIL'}", flush=True)
    lines = [ln for ln in out.splitlines() if ln.startswith(keep)] if keep else []
    for ln in lines:
        print(f"    {ln[:400]}")
    if not ok:
        tail = (out + "\n" + err).strip().splitlines()[-40:]
        print("    --- output tail ---\n    " + "\n    ".join(tail), flush=True)
    results[name] = {"rc": rc, "tracebacks": tb, "pass": ok, "lines": lines}
    return ok


def gpu_count():
    p = subprocess.run(["/usr/bin/python3", "-c", "import torch; print(torch.cuda.device_count())"],
                       cwd="/tmp", env=child_env(), capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return int(p.stdout.strip().splitlines()[-1])


def build(a, report):
    import glob
    import importlib.util

    vers = {d: md.version(d) for d in ("torch", "torchvision", "torchaudio", "triton", "vllm",
                                       "flashinfer-python", "flashinfer-cubin", "nvidia-cutlass-dsl",
                                       "apache-tvm-ffi", "cuda-python", "cuda-bindings", "nixl", "nixl-cu13",
                                       "ai-dynamo", "ai-dynamo-runtime", "pydantic", "pydantic-core",
                                       "nvidia-nccl-cu13", "nvidia-cudnn-cu13", "nvidia-cublas", "deep_ep")
            if _has(d)}
    report["versions"] = vers
    for k, v in sorted(vers.items()):
        print(f"VERSION {k} {v}")
    assert vers.get("flashinfer-python") == vers.get("flashinfer-cubin"), vers
    assert "+" in vers["flashinfer-python"], "flashinfer-python is not the locally built wheel"
    assert vers.get("ai-dynamo") == a.dynamo_version == vers.get("ai-dynamo-runtime"), vers
    assert vers.get("nixl-cu13") == a.nixl_version, vers
    assert vers.get("nixl", a.nixl_version) == a.nixl_version, vers
    if "nixl" not in vers:
        print("NOTE: no `nixl` meta package (NIXL build made no nixl-meta wheel)")
    assert not _has("nixl-cu12"), "nixl-cu12 installed"
    assert not os.environ.get("FLASHINFER_DSL_FMHA_LOCAL_DIR"), "FLASHINFER_DSL_FMHA_LOCAL_DIR is set"
    # FlashInfer artifact pin and the prefetched leaf.
    import flashinfer.artifacts as artifacts
    prefetch = json.load(open(a.prefetch_record))
    assert prefetch["artifact_path"] == artifacts.ArtifactPath.DSL_FMHA, prefetch
    manifest = [ln for ln in open(os.path.join(prefetch["leaf"], "checksums.txt")) if ln.strip()]
    so = glob.glob(os.path.join(prefetch["leaf"], "*.so"))
    assert prefetch["so_count"] == len(manifest) == len(so), (prefetch["so_count"], len(manifest), len(so))
    report["dsl_fmha"] = prefetch
    print(f"DSL_FMHA {prefetch['artifact_path']} {len(so)} .so in {prefetch['leaf']}")
    # NIXL EP build for the installed torch (upstream BUILD_NIXL=true), wheel-bundled UCX.
    import torch
    mm = "".join(torch.__version__.split(".")[:2])
    spec = importlib.util.find_spec("nixl_ep_cu13")
    ext = glob.glob(os.path.join(spec.submodule_search_locations[0], f"nixl_ep_cpp_torch{mm}.*")) if spec else []
    assert ext, f"no nixl_ep_cpp_torch{mm}"
    ucx = sorted(os.path.relpath(p, SITE) for p in glob.glob(os.path.join(SITE, "**", "libucp*"), recursive=True))
    report["nixl"] = {"nixl_ep": ext, "bundled_libucp": ucx}
    print(f"NIXL_EP {ext[0]}; bundled libucp {ucx}")
    # uv pip check: recorded (dynamo-deps.txt and the nightly Dynamo wheels are not one resolve).
    p = subprocess.run(["uv", "pip", "check", "--system"], capture_output=True, text=True)
    lines = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()]
    report["uv_pip_check"] = {"rc": p.returncode, "lines": lines}
    for ln in lines:
        print(f"UV_PIP_CHECK {ln}")
    # Import checks without a driver (vLLM logs a Traceback for the nixl_ep probe without libcuda).
    res = {}
    run_child("d_vllm_nixl_version_connector", CHECK_D % {"nixl": a.nixl_version}, res, False,
              keep=("CHECK_NIXL_VERSION_OK", "UCX_GUARD_OK"))
    for name, code in CHECK_E.items():
        run_child(name, code % {"native": False} if "%(native)" in code else code, res, False,
                  keep=("IMPORT_OK", "HAS_NIXL_EP", "UCX_GUARD_OK"))
    report["checks"] = res
    return all(r["pass"] for r in res.values())


def gpu(a, report):
    res = {}
    run_child("g1_torch", G1_TORCH % {}, res, keep=("TORCH", "ARCH_LIST", "DEVICE", "MATMUL_OK", "CONV_OK"))
    run_child("g2_nccl", G2_NCCL % {}, res, keep=("NCCL_OK",))
    names = []
    if a.names_file:
        names = [ln.strip() for ln in open(a.names_file) if ln.strip()]
    fmha_env = {"CUDA_VISIBLE_DEVICES": ""} if a.fmha_host_only else None
    run_child("g3_dsl_fmha_pinned_path", G3_FMHA % {"sm": a.sm, "names": names}, res,
              extra_env=fmha_env, keep=("FMHA_PIN", "FMHA_LOADED"))
    if names:
        run_child("g3b_dsl_fmha_whole_leaf", G3_FMHA % {"sm": a.sm, "names": []}, res,
                  extra_env=fmha_env, keep=("FMHA_PIN", "FMHA_LOADED"))
    run_child("a_torch_nixl_ep", CHECK_A, res, keep=("NIXL_EP_OK", "UCX_GUARD_OK"))
    run_child("b_loopback_dram", LOOPBACK % {"mem": "DRAM", "dev": 0, "peer": 0}, res,
              keep=("LOOPBACK_OK", "backend", "plugins", "UCX_GUARD_OK"))
    n = gpu_count()
    peer = 1 if n >= 2 else 0
    run_child("c_loopback_vram", LOOPBACK % {"mem": "VRAM", "dev": 0, "peer": peer}, res,
              keep=("LOOPBACK_OK", "backend", "UCX_GUARD_OK"))
    run_child("d_vllm_nixl_version_connector", CHECK_D % {"nixl": a.nixl_version}, res,
              keep=("CHECK_NIXL_VERSION_OK", "UCX_GUARD_OK"))
    for name, code in CHECK_E.items():
        run_child(name, code % {"native": True} if "%(native)" in code else code, res,
                  keep=("IMPORT_OK", "HAS_NIXL_EP", "UCX_GUARD_OK"))
    run_child("m_proc_maps", MAPS % {}, res, keep=("MAPPED", "UCX_GUARD_OK", "MAPS_OK"))
    report["checks"] = res
    return all(r["pass"] for r in res.values())


def _has(dist):
    try:
        md.version(dist)
        return True
    except md.PackageNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    ap.add_argument("--prefetch-record", default="/opt/q3vl-rubin/dsl-fmha-prefetch.json",
                    help="--build: the record prefetch_dsl_fmha.py wrote")
    ap.add_argument("--dynamo-version", help="--build: the expected ai-dynamo / ai-dynamo-runtime version")
    ap.add_argument("--nixl-version", default="1.5.0", help="the NIXL version vLLM requires")
    ap.add_argument("--names-file", help="--gpu: load only these export names (default: the whole leaf)")
    ap.add_argument("--sm", default="sm_107a", help="--gpu: the DSL FMHA SM leaf to load")
    ap.add_argument("--fmha-host-only", action="store_true",
                    help="--gpu: load the FMHA exports with no GPU visible (a host whose GPU is not --sm)")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    report = {"mode": "build" if a.build else "gpu"}
    try:
        ok = build(a, report) if a.build else gpu(a, report)
    except AssertionError as exc:
        print(f"ASSERTION: {exc}", flush=True)
        report["assertion"] = str(exc)
        ok = False
    report["pass"] = ok
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("VALIDATE_IMAGE_PASS" if ok else "VALIDATE_IMAGE_FAIL", report["mode"], flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
