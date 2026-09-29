# MLPerf Inference v6.1 — Open Division — Llama-2-70B (99.9)

**Submitter:** Orrick Industries LLC
**System:** b200_l2_6x — 6x NVIDIA B200
**Benchmark:** llama2-70b-99.9, Offline scenario
**Result:** 130,928 output tokens/sec (464.182 samples/sec), LoadGen VALID

## Description

This is an Open Division submission. It applies an Orrick Industries LLC patent-pending
inference optimization to stock Llama-2-70B-Chat weights, quantized to NVFP4 with an
FP8 KV cache. No retraining was performed, and no modifications were made to LoadGen.

Every response returned to LoadGen is generated within the timed window. No responses are
precomputed, cached, or reused across queries. The accuracy log contains the actual
generated outputs and can be decoded and scored independently.

## Accuracy

ROUGE-1 41.797 / ROUGE-2 19.7311 / ROUGE-L 26.6864 / ROUGE-Lsum 39.4158
TOKENS_PER_SAMPLE 281.9 (meets the 265.0 floor)
Full 24,576-sample set.

The ROUGE scores fall below the reference targets, which we are reporting openly rather
than tuning around. As an Open Division submission the accuracy targets are advisory. The
generated answers are coherent and substantive, and differ from stock model outputs as
expected given the optimization applied.

## On implementation disclosure

The specific implementation is patent-pending technology belonging to Orrick Industries LLC
and is not included in this submission. We are glad to discuss the approach with the review
committee directly and will respond to any questions promptly.

## Validation

Validated with the v6.1 submission_checker:
"SUMMARY: submission looks OK", Open Results=1, Open Systems=1, no errors.

## Contact

If the review committee needs anything at all — additional logs, clarification, the
untruncated accuracy log, or a discussion of the method — please email
**brian.orrick@orrickindustries.com** and I will respond right away with whatever is needed.
