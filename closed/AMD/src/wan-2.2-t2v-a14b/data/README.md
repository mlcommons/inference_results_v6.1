# `data/`

The harness expects two files at run time. Neither is checked in:

* `vbench_prompts.txt` – 248 VBench prompts, one per line. Used as the QSL.
* `fixed_latent.pt` – 9.6 MB BF16 initial-noise tensor of shape
  `[1, 16, 21, 90, 160]`. Used by the real backend to make generation
  deterministic across runs.

Both files live in the official MLCommons reference at
[`mlcommons/inference/text_to_video/wan-2.2-t2v-a14b/data/`](https://github.com/mlcommons/inference/tree/master/text_to_video/wan-2.2-t2v-a14b/data)
and are fetched by `tools/fetch_data.py`:

```sh
# Default: vbench_prompts.txt + fixed_latent.pt into <repo>/data/.
python3 -m tools.fetch_data
```

The downloader is stdlib-only (no external deps required), pinned to a
specific upstream commit by default for reproducibility, and verifies
each download against git's own blob SHA-1 so corruption is impossible
to miss.

Optional inputs (only needed for accuracy mode):

```sh
# Also fetch calibration_prompts.txt + samples_filename_ids.txt.
python3 -m tools.fetch_data --with-calibration --with-samples-list
```

Useful flags:

* `--commit <SHA|master>` – override the pinned commit. Pass `master` to
  always grab the current tip of `mlcommons/inference` (size check only,
  no blob-SHA verification).
* `--force` – re-download even when the file is already present at the
  expected SHA. Use after a corrupted run.
* `--data-dir <PATH>` – write somewhere other than `<repo>/data/`.

For the Mock dry-run neither file is required: the loader falls back to
a synthetic 32-prompt set with a warning.

For the real WanBackend smoke test (`scripts/smoke_wan22.sh`) we ship a
small set of realistic prompts in [`synthetic_prompts.txt`](synthetic_prompts.txt).
These are short, diverse text-to-video prompts kept in the repo so the
smoke test can run before the official VBench list has been downloaded.
