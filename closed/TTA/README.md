# MLPerf&trade; Inference v6.1 &mdash; TTA Submission

TTA's closed-division submission for Llama 3.1 8B, running on a
**ManyCoreSoft DeepGadget dg5W** with 4&times; NVIDIA RTX PRO 6000 Blackwell Server Edition.

The implementation is NVIDIA's TensorRT-LLM MLPerf harness, run inside NVIDIA's official
MLPerf Inference container. TTA's contribution is the system configuration
(`src/configs/`) and the calibration setting described in
[Quantization &amp; Calibration](#quantization--calibration).

---

## Results

| Benchmark | Scenario | Throughput | Latency (p99) | Accuracy | LoadGen |
|---|---|---:|---|---|---|
| llama3.1-8b | Offline | **24,511.8** tok/s | &mdash; | ROUGE PASS | VALID |
| llama3.1-8b | Server  | **24,510.5** tok/s | TTFT 1,512 ms<br>TPOT 85.0 ms | ROUGE PASS | VALID |

Server constraints are TTFT &le; 2,000 ms and TPOT &le; 100 ms &mdash; both satisfied with margin.
Compliance test **TEST06 PASS** for both scenarios.

### Accuracy detail

| Metric | Offline | Server | Threshold (99% of reference) |
|---|---:|---:|---:|
| ROUGE-1 | 38.5238 | 38.5242 | &ge; 38.3914 |
| ROUGE-2 | 15.9869 | 15.9887 | &ge; 15.7484 |
| ROUGE-L | 24.3988 | 24.3995 | &ge; 24.2507 |
| ROUGE-Lsum | 35.6579 | 35.6582 | &ge; 35.4351 |
| gen_len | 8,178,576 | 8,178,682 | &ge; 7,350,880 |

All 13,368 samples evaluated.

---

## System

| | |
|---|---|
| **System** | ManyCoreSoft DeepGadget dg5W (SKU2-T1), GENOAD8X-2T/BCM mainboard |
| **Accelerators** | 4&times; NVIDIA RTX PRO 6000 Blackwell Server Edition &middot; 96 GB GDDR7 &middot; 600 W TGP |
| **Accelerator link** | PCIe Gen5 x16 per GPU &mdash; no NVLink |
| **Host CPU** | 1&times; AMD EPYC 9124 (16C / 32T) |
| **Host memory** | 512 GB DDR5 ECC (8&times; 64 GB, DDR5-5600 @ 4800 MT/s) |
| **Storage** | 2 TB NVMe SSD (xfs) |
| **Cooling** | Direct liquid cooling (DLC) |
| **OS** | Rocky Linux 9.7 |

Full machine-readable description:
[`systems/DeepGadget_dg5W_RTXPro6000_96GBx4_TRT.json`](systems/DeepGadget_dg5W_RTXPro6000_96GBx4_TRT.json)

### Software stack

| Component | Version |
|---|---|
| TensorRT | 10.14.1.48 |
| CUDA | 13.1 |
| cuDNN | 9.17 |
| NVIDIA Driver | 595.45.04 |

---

## Running the benchmark

### Prerequisites

Everything runs inside NVIDIA's official MLPerf Inference container, which carries the
harness, TensorRT, and TensorRT-LLM:

```
nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

You also need the MLCommons-provided Llama 3.1 8B weights, the CNN/DailyMail evaluation
set, and the official 1,000-sample calibration set. Follow NVIDIA's setup for
`make download_model` / `make download_dataset` / `make preprocess_data`.

> LoadGen must be the pristine v6.1 build. A patched LoadGen stamps
> `Loadgen built with uncommitted changes!` into the run log, which the submission
> checker rejects.

### Steps

Run **Offline first** &mdash; Server reuses the same quantized checkpoint and skips step 2.

#### 1 &middot; Install the config

```bash
cp -r src/configs/DeepGadget_dg5W_RTXPro6000_96GBx4_TRT configs/
export SYSTEM_NAME=DeepGadget_dg5W_RTXPro6000_96GBx4_TRT
```

#### 2 &middot; Quantize to NVFP4

The engine builder needs `transformers==4.57.1` while ModelOpt needs `<=4.56`, so
quantization runs standalone and the harness then picks up the checkpoint it leaves behind.
Delete any stale checkpoint first &mdash; the harness silently reuses one if it exists.

```python
# calibrate on the full 1,000-sample official set at sequence length 2560
tok = AutoTokenizer.from_pretrained(MODEL_DIR, model_max_length=2560, use_fast=True)
enc = tok(ds["text"][:1000], return_tensors="pt", padding=True,
          truncation=True, max_length=2560).to(model.device)

cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)          # NVFP4 weights + activations
cfg["quant_cfg"].update(mtq.FP8_KV_CFG["quant_cfg"])  # FP8 KV cache
model = mtq.quantize(model, cfg, forward_loop=loop)

export_tensorrt_llm_checkpoint(model, "llama", torch.float16, export_dir=CKPT,
                               inference_tensor_parallel=1, inference_pipeline_parallel=1)
```

Run it under `transformers==4.55.0`, then restore `4.57.1` before building.

#### 3 &middot; Build the engine and run

```bash
make run_harness      RUN_ARGS="--benchmarks=llama3.1-8b --scenarios=Offline --test_mode=PerformanceOnly --min_duration=600000"
make run_harness      RUN_ARGS="--benchmarks=llama3.1-8b --scenarios=Offline --test_mode=AccuracyOnly"
```

Repeat with `--scenarios=Server`.

> `min_duration=600000` is not optional for Server. TTFT grows with run length as the
> queue builds, so a short sweep overestimates the sustainable QPS &mdash; 191 passes at
> 180 s but fails at 600 s. 190 is the highest QPS that holds for the full 600 s.

#### 4 &middot; Compliance and staging

```bash
make run_audit_test06 RUN_ARGS="--benchmarks=llama3.1-8b --scenarios=Offline"
make stage_results
```

#### Expected result

A single run lands **VALID at ~24.5k tok/s** with all ROUGE metrics passing and TEST06
PASS. The submitted figures are the best of repeated engine builds &mdash; `multiple_profiles`
samples different kernel tactics each build, giving a &plusmn;0.8% spread across the
24.48&ndash;24.51k band. The configuration is identical in every build.

### Key configuration

Both scenarios share the engine build flags; `gpu_batch_size` and the LoadGen target differ.

| Setting | Offline | Server |
|---|---|---|
| `precision` | fp4 (NVFP4) | fp4 (NVFP4) |
| `kv_cache_dtype` | fp8 | fp8 |
| `tokens_per_block` | 64 | 64 |
| `max_num_tokens` | 4096 | 4096 |
| `max_input_len` / `max_seq_len` | 2540 / 2668 | 2540 / 2668 |
| `gpu_batch_size` | 512 | 2048 |
| `kvcache_free_gpu_mem_frac` | 0.95 | 0.95 |
| LoadGen target | `offline_expected_qps: 360` | `server_target_qps: 190` |

Full configs: [`src/configs/`](src/configs/DeepGadget_dg5W_RTXPro6000_96GBx4_TRT)

---

## Quantization &amp; Calibration

Weights and activations are quantized to **NVFP4** with an **FP8 KV cache** using NVIDIA
ModelOpt, following the scheme in [`documentation/calibration.md`](documentation/calibration.md).
No retraining is performed and only the MLCommons-provided calibration set is used.

Calibration is performed by invoking ModelOpt directly rather than through the harness'
`make calibrate` target, because the builder and the quantizer require different
`transformers` versions in this stack. The quantized checkpoint is produced first, and
the harness then builds the engine from it. Since the checkpoint already exists at build
time, the harness' own quantization step is a no-op &mdash; the `trtllm_checkpoint_flags`
entries in `src/configs/` are not consulted along this path. The quantization math is
unchanged.

The calibration actually used:

| Parameter | Harness default | This submission |
|---|---|---|
| calibration samples | 512 | **1000** &mdash; the full official calibration set |
| calibration sequence length | 512 | **2560** |

A sequence length of 512 truncates the calibration articles far below the 2,540 tokens
the model sees at inference time, which skews the activation scales. Aligning the two
recovers ROUGE-2 from 15.39 (below the 15.75 threshold) to 15.99.

---

## Submission layout

```
closed/TTA/
├── README.md
├── documentation/            # calibration, bandwidth, harness commands
├── results/
│   └── DeepGadget_dg5W_RTXPro6000_96GBx4_TRT/
│       └── llama3.1-8b/
│           ├── Offline/      # performance, accuracy, TEST06
│           └── Server/
├── src/
│   ├── configs/              # TTA system configuration
│   └── llama3_1-8b/          # benchmark sources
└── systems/                  # system description JSON
```

## Validation

Verified with the official MLCommons v6.1 submission checker:

```bash
python3 tools/submission/submission_checker/main.py \
        --input <submission_root> --submitter TTA --version v6.1
```

```
Closed Results=2, Closed Systems=1
SUMMARY: submission looks OK
```

No `--submission-exceptions` required.

---

<sub>MLPerf&trade; is a trademark of MLCommons&reg; Association. Unauthorized use is strictly prohibited.</sub>
