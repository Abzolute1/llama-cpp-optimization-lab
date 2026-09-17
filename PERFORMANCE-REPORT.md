# Local inference optimization — performance report

These documents describe optimization methods and historical measurements for two models. Personal paths, hardware identifiers and application-specific records are omitted. Results are hardware-dependent. Selected source patches, portable build tools and numerical summaries are now included; see START-HERE.md. Private original workloads and historical binaries are excluded. The new synthetic harness is not a replay of the historical benchmarks.

## Qwen3.8 Flash-Next performance investigation

**Recommendation: retain the original default and keep these as optional acceleration configurations.** Most of the native editing gain comes from stock ngram drafting on the original runtime (1.94×), without the source candidate’s large private-memory cost. The combined candidate is useful for a model kept resident through long inputs and editing, but is not an across-the-board upgrade: it commits about 34 GiB of private anonymous host memory, slows repetitive rewriting, and took longer to load.

### Model and baseline

Model: Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64, with unchanged model weights.

- Original runtime: Unsloth b10715-mix-86bd2d3, build 10715, source `92cedc8679d145902ead3f006258e8672eac11e6`. Runtime reports 0.3.0-dev. compatible CPU and CUDA toolchains, multi-architecture CUDA build including the target GPU architecture; selected CPU backend haswell.
- Original executable: `<LOCAL_PATH>`.
- Requested context 90,000 (allocated 90,112), Flash Attention on, Q8_0 K/V, one slot, batch/microbatch 1,024/1,024, 12 decode/24 batch threads with strict 0–11/0–23 affinity, poll 100, CPU MoE 42, automatic GPU layer offload, fit off. No speculative drafting was configured originally.

```text
<SERVER_EXECUTABLE> --model <MODEL_FILE> --alias Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64 --host <LOOPBACK_ADDRESS> --port <BENCHMARK_PORT> --flash-attn on --ctx-size 90000 --cache-type-k q8_0 --cache-type-v q8_0 --no-webui --jinja --parallel 1 --fit-target 1024 --batch-size 1024 --ubatch-size 1024 --threads 12 --threads-batch 24 --cpu-range 0-11 --cpu-strict 1 --cpu-range-batch 0-23 --cpu-strict-batch 1 --poll 100 --slot-prompt-similarity 0 --no-mmproj --fit off --n-gpu-layers auto --kv-unified --n-cpu-moe 42 --metrics --verbosity 2
```

Configure CUDA and shared-library search paths for the selected build. No experimental GGML/OMP overrides were inherited by the baseline. See the captured environment for exact values.

### What “n-gram” means in this model

Flash-Next’s learned PLE subsystem performs bigram/trigram-dependent table lookup as part of the target model computation. Its Q5_1 table occupies 35.763 GiB and emits 16 rows/token. This is separate from the runtime’s optional ngram-mod speculative drafting. PLE has no draft acceptance rate. Runtime speculation proposes tokens, then uses the unchanged target model to verify them. An additional existing Q8 MTP draft head was evaluated separately.

The GGUF filename is not an accurate inventory of tensor quantization. Actual tensors include 48 MXFP4 down-expert tensors, 72 IQ1_M and 24 IQ2_S gate/up-expert tensors, 666 Q8_0 tensors, 388 F32 tensors, 24 BF16 tensors and one F16 tensor, plus the Q5_1 PLE table. No IQ4_XS tensor is present. Quantization types were read directly from the GGUF headers. Cache eligibility for existing IQ1_M experts changes placement only; it does not re-quantize weights.

### Baseline measurements

Three repetitions per workload, 24 native inference requests in baseline-01. Fresh prompt cache is disabled except the explicitly warmed conversation fixture. Fixed greedy sampling: temperature 0, top_k 1, top_p 1, min_p 0, repeat_penalty 1, seed 777. Frozen prompts and token IDs live in `prompts.json`. Prompt-cache warmth and OS file-cache warmth are distinct. The first review had slower page-cache behavior; the table reports all-three medians, not its best run.

| Workload | Input tokens | Prefill tok/s | Decode tok/s | TTFT s | Response s |
|---|---:|---:|---:|---:|---:|
| A: fresh long ingestion | 21308 | 305.46 | 25.97 | 69.762 | 77.978 |
| B: code review | 6250 | 302.03 | 27.99 | 20.696 | 29.781 |
| C: exact source edit | 1006 | 296.56 | 29.02 | 3.393 | 35.679 |
| D: ordinary prose | 74 | 61.12 | 29.19 | 1.211 | 9.946 |
| E: tool JSON fixture | 70 | 58.28 | 28.21 | 1.202 | 2.974 |
| F: cached conversation | 4243 | 66.56 | 28.36 | 0.978 | 9.991 |
| G: repeated code rewrite | 500 | 193.23 | 29.09 | 2.588 | 17.577 |
| H: free-form reasoning | 110 | 76.29 | 29.07 | 1.443 | 10.189 |

Conversation prefill speed measures only the newly evaluated suffix, not all 4,243 tokens. The short tool JSON fixture is supplemented by native function-call validation. Native edit outputs commonly omit the final newline; content agreement is not strict file-byte correctness. Short 256-token screens do not satisfy full 500-word review/prose requests; those require the final quality pass.

Initial observed model load: 85.443s; this was not a cold-boot/drop-caches test. Later original warm loads were around 3.4–5.4 s. Baseline peak total GPU memory: 14753MiB, including desktop use. Process RSS is predominantly reclaimable mmap pages; it must not be equated with private committed RAM. Raw CPU/GPU utilization, clocks, power, RAM, faults and I/O are preserved per request.

### Measurement interpretation

- Compare native prefill and common decode rates separately. Original native decode counts predicted_n−1; the older fork counts predicted_n. `RUNS.md` normalizes both to `(predicted_n−1)/predicted_ms`. Raw timings remain unchanged.
- Resource samples occur about every 0.5 s. CPU utilization 100% means one logical CPU. Early VRAM readings include desktop use; later runs also record process-specific NVML memory.
- Instrumented CUPTI/perf results locate bottlenecks and are excluded from speed comparisons. The stale-marker build-overlap arm and interrupted CPU-flat arm are explicitly excluded in `invalid-runs.json`.
- Same-file experiments keep model, quantization, context, sampling and other non-variable settings fixed. The optional higher-precision alternative is a separate model comparison.

### Measured bottlenecks

Original review prefill moved roughly 161 GiB of host-to-device data for 6,250 input tokens. CUPTI summed H2D copies to 10.10 s and the host cudaMemcpyAsync API to 13.34 s. These overlapping instrumented activities are not additive wall-time components. IQ1_M dequantization had 85,268 launches but only 0.506 s summed GPU duration. Transfer/synchronization work therefore deserves priority over a kernel rewrite based on launch count alone.

Original decode heavily exercised CPU expert dot products and OpenMP waiting while GPU utilization was around 42%. The measured CPU sample shares included MXFP4/IQ1_M/IQ2_S arithmetic and substantial waiting. Prefill and decode therefore need different solutions; making larger GPU batches fit by putting more experts in RAM can help one and harm the other.

### Experiments and current status

`RUNS.md` is the complete generated run catalog, with separate workload medians and ranges, failures, exclusions and configuration links. `STATUS.md` records checkpoints. `kernel-opportunities.md` and `turbo-source-review.md` explain the source hypotheses. `REPRODUCTION-GUIDE.md` describes the protocol. All source overlays remain in separately named experiment directories.

#### Experiment families and rejected alternatives

The complete run catalog retains individual configurations, raw results, failures and exclusions. These historical screens explain which paths were carried forward; their values are not substitutes for the final balanced comparison.

| Family | Finding and consequence |
|---|---|
| CPU threads, affinities, polling and OpenMP | Fewer decode threads, passive waiting and poll=0 generally lost speed. Twelve verification threads were useful for longer ngram drafts; six verification threads across one chiplet were slower. Lower CPU utilization alone was not considered a win. |
| Batch size and expert placement | Original batch 2048/CPU 44 reached about 411 review-prefill tok/s, with a small decode loss; CPU 42 at 2048 exhausted VRAM. Larger staged batches reached about 469 prefill but only about 21 decode. The optional original 1536 follow-up was deferred; the final candidate retains batch 1024 and CPU 42 so it does not claim a batch-size gain. |
| Newer builds and CUDA graphs | Fresh Unsloth/upstream GPU-specific builds did not automatically beat the original. Disabling CUDA graphs reduced decode to about 23–25 tok/s. Keep graphs enabled. |
| CPU arithmetic/scheduling kernels | Paired MXFP4 rows improved a microbenchmark by about 6% without a useful application gain. Flat CPU scheduling produced only a small newer-build screen gain; the Turbo variant was slower in real inference despite exact synthetic output comparisons. Microbenchmarks alone did not justify either change. |
| Host registration and staging | Read-only registration failed with driver error 801. Private writable mappings preserve file contents and support registration. They improved prefill but increase memory commitment and startup cost. Bounded staging avoids pinning all experts but has shown workload-dependent results. |
| Fixed expert pools | Many pool sizes slowed decode to roughly 18–26 tok/s. An efficient 80-slot pool reached about 30 on prose without a general win. These bundles expand context-parameter ABI and cannot be mixed with ordinary libllama/common libraries. |
| Turbo dynamic cache | Cache budget, eligible expert size, placement, overlap, asynchronous fills, churn and fusion settings trade prefill, prose, edit speed and VRAM. Lowering the eligibility threshold moves existing IQ1_M bytes; it does not change quantization. Inspect actual allocated capacity rather than requested capacity. |
| Runtime ngram drafting | Longer drafts substantially help source editing/copying; repetitive rewrites can regress. Acceptance is not itself a speed metric. Recurrent snapshots and adaptive backoff are separately evaluated rather than assumed to improve the same workloads. |
| Existing Q8 MTP head | At matched CPU 46, depth 2/depth 4 improved edit decode from about 27.15 to 32.91/37.09 tok/s, while prose fell from 27.20 to 26.71/22.15 and reasoning also slowed. It is less promising than ngram drafting for the measured editing task. |
| Masked attention | The synthetic sparse case improved 3.443× with 20 exact fixture outputs. Real long-context decode improved about13%; fresh response latency improved much less. Real affected-path numerical checks and task quality are reported separately. |
| Whole-expert transfers | Avoiding selected-ID readback slowed unpinned review prefill in both process orders. With private pinned mappings it was approximately tied with selected-expert transfers, not a substantial additive gain. |

Turbo's current cache path uses 64-bit logical-row masks. Ten experts per token means long speculative verification batches exceed that path and use the stock bypass. Raising the limit beyond 64 without redesigning the masks would be incorrect. This helps explain why more cache hits or longer drafts do not automatically translate into better verification throughput.

#### Results since the authorized reboot resume

| Experiment | Matched control | Experimental result | Interpretation |
|---|---|---|---|
| 84,859-token retrieval, masked attention | Newer base decode 20.59/20.65 tok/s | 23.27/23.37 tok/s | About 13% faster decode; only about 0.9% lower fresh response time because prefill dominates |
| Source edit, fixed ngram16 with 12 verification threads | No drafting 26.72/28.94 tok/s | 51.04/48.32 tok/s | Substantial repeated improvement on copying most of a source file |
| Source copying, same drafting setup | 26.96/25.65 tok/s | 59.44/61.42 tok/s | More than twice the measured decode rate in both passes |
| Repetitive rewrite, same drafting setup | 26.99/26.47 tok/s | 23.56/19.17 tok/s | Draft verification is unprofitable on this workload |
| Repetitive copying, same drafting setup | 27.09/28.65 tok/s | 34.20/36.65 tok/s | Some partially accepted drafts still save time; acceptance alone is an inadequate policy signal |

All six long-context retrieval runs (original, newer base, masked candidate, each twice) passed strict JSON and exact factual checks. The candidate's mean TTFT was 297.91 seconds versus 299.62 for the newer base and 306.42 for the original. These are fresh-context requests, not cached chat latency. See long-context results and quality checks.

Across the six no-draft, fixed-draft and adaptive-draft process runs, each of the five workloads produced identical output bytes in every arm. Source edits still omitted the requested final newline, and repetitive rewrites still used an unwanted outer fence in every arm. These shared defects remain failures of strict formatting; content agreement does not erase them. See drafting quality checks.

The first adaptive policy (pause for 1,024 tokens after fewer than 85% of 32 proposed tokens are accepted) recovered only part of the repetitive-rewrite loss. It also stopped too early on profitable repetitive copying. It is not ready as a general automatic default.

Parallel private-page preparation did not help: median load was 22.631 seconds with six workers and 21.109 seconds with twelve, compared with 20.627 seconds with none. Subsequent combinations therefore use no extra workers. See all nine load samples.

The warm-file figure must not be generalized to ordinary restarts. In the subsequent pinned-plus-bulk comparison, `cow-bulkcopy-bulk-0` took 67.060 seconds to load without explicit prewarming after another pinned process exited. Its corresponding first control had taken 23.755 seconds from a different inherited file-cache state. Pinning can leave fewer source pages cached for the next process. These differing load samples are retained; the bulk-transfer flag acts during inference and should not be credited or blamed for the cache-state load difference.

Numerical checks require care: prose/edit fixed-prefix comparisons below the masked-attention activation threshold were identical, but they did not exercise the patch. A separate 6,250-token review diagnostic did exercise it. Candidate versus control agreed on 62/64 greedy positions; a second unmodified control agreed with the first on 63/64. Reported top-32-plus-residual distributions also varied within the control. This does not establish bit-exact real-model equivalence or isolate every change to the patch. The earlier exact synthetic kernel fixtures and successful real retrieval checks are separate evidence. See affected-path comparison and control repeat.

#### Candidate selected for focused validation

`R/closeout-candidate.json` uses `bundle-closeout-cow-mask`: the screened private-mapping loader and masked-attention CUDA library, with stock ngram16 drafting and twelve verification threads. The primary model, 90k context, Q8 K/V, batch 1024 and CPU 42 placement remain unchanged. The assembly checks matching public ABI headers and records every library hash in the assembly manifest.

This combination targets the three demonstrated opportunities: prompt processing, long-context decoding, and source copying/editing. Its combined behavior is measured in the final comparison below. Adaptive backoff, recurrent snapshot extensions, whole-expert copying, staging and the Turbo runtime are not part of this candidate. The comparison includes both the untouched original and the same stock drafting configuration on the original runtime, so the source changes must earn their additional complexity and startup cost.

### Final full-answer comparison

Two reversed process orders, one request per workload in each process. Values are means of the two observations; the machine-readable summary retains samples and ranges. Decode uses the common token-count convention. These are bounded measurements, not confidence intervals or a universal speedup.

| Workload | Original decode tok/s | Drafting only | Combined candidate | Original response s | Drafting only s | Candidate s |
|---|---:|---:|---:|---:|---:|---:|
| review | 26.76 | 26.88 | 26.96 | 75.14 | 73.36 | 63.70 |
| edit | 27.25 | 52.87 | 56.12 | 38.15 | 21.14 | 19.11 |
| prose | 27.92 | 28.30 | 29.26 | 25.94 | 25.54 | 26.23 |
| tool | 26.40 | 27.38 | 28.17 | 3.19 | 3.00 | 2.53 |
| conversation | 27.50 | 27.29 | 26.53 | 66.53 | 67.09 | 68.97 |
| repetitive | 26.76 | 22.43 | 22.97 | 19.03 | 22.09 | 20.78 |
| reasoning | 26.92 | 27.84 | 28.32 | 60.77 | 58.39 | 44.50 |

Review prefill was 293.89 → 281.23 → 386.68 tok/s for original, drafting only and combined candidate, respectively. The corresponding TTFT means were 21.28 s / 22.24 s / 16.17 s. The source candidate therefore improves review prompt processing by about 32%, while review decode is essentially unchanged. Editing decode improves about 94% through configuration alone and about 106% with the combined candidate; the latter adds about 6% over drafting only.

Full-answer wall time must be interpreted alongside output length. Candidate review answers were 787/841 words versus 927/938 for the original; candidate reasoning was 909 words versus 1,040. Prose was 577 versus 517 words. The six cached-conversation answers all hit the 1,800-token cap, so they do not establish full-answer completion.

#### Final near-limit context comparison

The last pair used the same 84,859-token prompt, unchanged 90,000 requested context, and the same 203-token answer. Both passed strict JSON and every expected field.

| Metric | Original | Combined candidate |
|---|---:|---:|
| Prefill tok/s | 281.81 | 395.38 |
| Decode tok/s | 20.47 | 23.32 |
| First output s | 301.14 | 214.65 |
| Complete response s | 311.01 | 223.31 |

The combined candidate reduced the request from 311.01 to 223.31 seconds (28.2%), with about 40% faster prefill and 14% faster decode. Unlike the earlier mask-only test, the private pinned loader also reduces the dominant prefill time. This final combination has one long-context pair; the six earlier runs separately support the masked-attention component’s decode benefit. It is not a broad long-context quality benchmark. See final long-context quality and raw summary.

#### Startup and memory in the final full-answer runs

At readiness, original private anonymous memory was about 380 MiB and drafting-only about 395 MiB, versus 34,804–34,850 MiB for the candidate. Total RSS was similar because original memory was predominantly file-backed. Candidate process swap was approximately 55–99 MiB at readiness; the original/control samples showed zero. CUDA-registered memory is not represented as Linux `Locked` bytes in these smaps records. Peak process-specific GPU memory was 13,678 MiB original, 13,718 MiB drafting-only and 13,440 MiB candidate. This small VRAM reduction does not offset the substantial change in host-memory commitment.

#### Answer quality

All six source-edit outputs were byte-identical and correct after trimming outer whitespace, but omitted the required final newline. All six repetitive rewrites were also byte-identical and content-correct after stripping the unwanted fence. Both remain strict-formatting failures. The tool-shaped JSON fixture parsed in all six runs but differed from the frozen expected schema; this fixture has a documented wording/schema ambiguity. Separately, all six native function-call smoke checks passed, while their direct-answer JSON checks failed because of an outer Markdown fence.

The six code reviews cover the main architecture but overstate some guarantees and omit important custody/readiness details. Both original and candidate prose omit important cancellation/committed-effect qualifications. Both crash-recovery designs have substantive durability and replay flaws; the candidate includes internally inconsistent recovery rules. The measured execution gains use the same stored model, but this small sample does not establish unchanged reasoning quality or bit-exact generation. Review answers also differed between repetitions within each arm. No prompt or scoring criterion was altered to rescue a failure. See mechanical checks, predeclared rubric and answer-specific manual assessment.

### Revisions under test

| Family | Pinned source revision |
|---|---|
| Original Unsloth |92cedc8679d145902ead3f006258e8672eac11e6|
| Newer Unsloth b10909-mix-bea84f7 |329b6160f513915f1c607dbfae3d5ce864a64a4f|
| Upstream llama.cpp |97e4ca73582084f2751767f80c86237483ecc381|
| TheTom TurboQuant fork |407f3237bfb3eeaff61546797de3d8c1a96be748|

The newer source families were built separately for the target GPU architecture. Runtime build strings from archive builds may lack git metadata; the pinned source hashes and build manifests are authoritative. The old production source/installations were never used as mutable experiment trees.

### Reproduction

See REPRODUCTION-GUIDE.md for the methodology. Source patches, compatible build inputs and a separately reviewed evidence package are required to repeat the exact experiment.


---

## Qwen3.6-35B-A3B performance engineering

### Measured outcome

| Workload | Input tokens | Prefill tok/s, original → selected | Decode tok/s, original → selected | TTFT seconds, original → selected |
|---|---:|---:|---:|---:|
| Short review | 957 | 399.6 → 973.6 | 67.13 → 87.56 | 2.436 → 1.025 |
| Inventory review | 9,154 | 390.9 → 2,361.0 | 64.56 → 76.15 | 23.461 → 3.922 |
| Exact 25-field extraction | 9,051 | 398.1 → 2,366.3 | 64.16 → 79.27 | 22.779 → 3.872 |
| source-code review | 6,274 | 433.9 → 2,361.0 | 66.04 → 87.53 | 14.504 → 2.701 |
| Exact source-file edit | 1,015 | 443.6 → 1,017.3 | 66.39 → 197.38 | 2.318 → 1.036 |
| Twelve-fact extraction | 6,445 | 425.7 → 2,389.6 | 64.99 → 72.65 | 15.182 → 2.723 |

Values are medians from **five candidate runs and three interleaved original anchors**, including three paired main-workload rounds; ranges, every raw value, client-side decode estimates and paired changes are in final-confirm-summary.json. Both arms used the same frozen inputs, original weight/KV precision, context and sampling. Ordinary reviews were capped at 512 tokens. Exact extraction and editing passed every run. Both arms retained the same 11/12 fact score and the same known wrong field.

The sequential 81,582-token extraction improved from 359.99 to 2,410.42 prefill tokens/sec and 226.678 to 33.901 seconds TTFT. Its decode 48.89→63.09 includes possible reuse of n-grams from the earlier related extraction; it is not the standalone long-context decode claim. The current cohort has one new original long anchor and three candidate runs; the unchanged original also has three earlier repetitions at 359.80 prefill / 49.68 decode medians. Standalone first-request long-context results are reported separately.

#### Resource interpretation

The selected file-backed loader reduced non-file-backed memory while increasing mapped-file RSS. Higher RSS alone does not imply greater committed memory. Compare anonymous memory, available RAM, and matched VRAM peaks. Generation power and tiny-request latency can regress even when throughput improves.

### What was preserved and measured

The original executable is `<LOCAL_PATH>`, version 10530, commit `bd1bf025fc55ffa1fcb2ba6d8bb8805f35671d1f` from 19 August 2026. This is [TheTom's TurboQuant fork](https://github.com/TheTom/llama-cpp-turboquant/tree/bd1bf025fc55ffa1fcb2ba6d8bb8805f35671d1f), including its own experimental MoE cache. It is not an unmodified official upstream build.

The original Release build uses a compatible compiler/toolchain, the target GPU architecture, native CPU instructions, CPU repacking, OpenMP, and CUDA graphs. BLAS, forced MMQ, and forced cuBLAS are off. See original build configuration and original source identity. Experimental CUDA compilation retained the target GPU architecture and used a compatible NVCC host compiler; individual build plans retain exact commands.

The model under test is Qwen3.6-35B-A3B UD-Q4_K_XL, with unchanged mixed-quantization weights. The architecture has 40 blocks and eight active routed experts per token.

The existing BF16 vision projector was loaded for comparable memory pressure in the performance tests. Its file is 902,822,016 bytes, SHA256 `e5c205cec2fd28f66c3895e4040021ab994b860323c3db8531640305ff49b322`. Primary performance requests are text, so these results do not establish image-encoding throughput.

### Original runtime configuration

The original requested context is 90,000 tokens, rounded to 90,112 by the server. Both KV caches are q8_0; Flash Attention is on. Batch 256, micro-batch 128, 12 decode threads, 12 batch threads, one slot, GPU layers 99, first 20 layers' routed experts on CPU, unified KV, fit disabled, fit target 512, load-mode none, Jinja enabled, reasoning preservation, and GPU projector offload. The original also requests priority 2, poll 100 and strict CPU ranges 0–11 for both thread groups.

The CPU-range/strict/priority/poll arguments are **not effective thread-pool controls in this server path**: it does not attach the corresponding pools. The OpenMP backend uses the thread counts. `--poll 0` therefore cannot receive credit for an apparent speed gain. Explicit GNU OpenMP affinity was separately tested and did bind threads, including inherited server/CUDA threads, but did not produce a repeatable improvement. See the source review, saved thread-placement snapshots, and GNU's [GOMP_CPU_AFFINITY documentation](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fCPU_005fAFFINITY.html).

### Benchmark discipline and practical limits

All ranked results use actual HTTP inference with the original GGUF, saved prompt token IDs, greedy sampling, seed 777, top_k 1, top_p 1, min_p 0 and repeat penalty 1. The six main workloads cover a 957-token review, 9,154-token inventory/review, 9,051-token exact 25-field JSON extraction, 6,274-token source-code review, 1,015-token exact source-file edit, and 6,445-token twelve-fact extraction. The long-context input is 81,582 tokens and requires 25 exact fields from the last of nine snapshots. Prompts, hashes and expected answers are in prompts.json.

The final review cap is 512 generated tokens; exact JSON has a 768-token cap and exact editing 2,048. The edit normally emits 938 tokens. Early exploratory reviews used 256 tokens; only matched caps are compared directly. Review excerpts are intentionally capped and are not complete software-development evaluations. Structured checks test correctness on these tasks, not every capability of the model.

The earlier CPU40 acceptance cohort has five paired fresh-server comparisons with alternating arm order. Final CPU32 confirmation uses five fresh candidate servers and three interleaved unchanged-original anchors: main workloads have three paired rounds plus two additional candidate rounds. The long sequential workload has one new paired original anchor and three candidate repetitions, with three earlier unchanged-original repetitions retained separately. Main-workload order is identical within each paired round; cohort counts are reported explicitly. Post-sequence API results with different preceding long-workload histories are retained as diagnostics; a dedicated matched fresh-API cohort provides the small-request comparison. This is crucial for n-gram speculation: repeating a deterministic answer in one process teaches its table that answer, independently of KV-prefix caching. Warm repeated-answer speeds are retained as diagnostics, never advertised as fresh prose speed. Each main workload occurs once per fresh process, but related workloads in that sequence can still share n-grams: the long snapshot extraction resembles the earlier short extraction. Its sequential decode result is labeled accordingly. A separate first-request long-context cohort excludes any earlier real answer from that process. The cold-loading cohort also starts with the exact edit before any other answer. The final runner, exact plans, per-request payloads/events, telemetry, launch environments, loaded-library paths and raw outputs remain available in the evidence and experiment directory.

Resources were sampled every 0.5 seconds: aggregate process CPU utilization (100% = one core), GPU utilization, board power, clocks, temperatures, global VRAM, process RSS/high-water mark, system available RAM and swap. Later runs include per-request `smaps_rollup`. NVML PCIe samples are short counter windows, not full-request byte integrals. GPU utilization does not measure achieved FLOPs; CPU polling can consume cores without useful work.

No compiler, linker, CPU microbenchmark, backend fixture, or large hash was allowed alongside ranked inference. Seven early affinity runs were found to overlap heavy work, explicitly excluded and rerun cleanly. A stalled draft run and instrumented routing runs are also excluded. timing-exclusions.json is part of the evidence, not an optional filter. Native newer-upstream decode timing uses a different numerator; cross-build summaries therefore also provide `(generated_tokens−1)/(wall−TTFT)`.

### The bottlenecks and changes that mattered

The original small prompt batches repeatedly move selected CPU expert weights across PCIe. Instrumented 9k-prompt execution spent most observed time waiting for CUDA completion; that wait includes compute and transfers, so it must not be called pure PCIe time. Measured H2D throughput was about 18.34 GiB/s at the original batch configuration. Larger micro-batches amortize weight transfers over more known prompt tokens. They do not make autoregressive generation parallel. Simply moving more experts into RAM to fit huge batches raised prefill above 3,100 tokens/sec but dropped decode to about 54; that tradeoff was rejected as the overall solution. See profile summary.

Decode is sensitive to host expert work and cache residency. The fork already contains GPU expert caching, but its default placement leaves it dormant. Allocating an explicit cache while moving more static experts to host memory gave a useful path forward. The original policy protected entries after only a few uses and did not age that protection; after switching tasks, generation could collapse even while old entries remained protected. Setting the existing hot-use threshold to 1,073,741,824 produces effectively LRU replacement for realistic run lengths, retaining the existing admission and replacement-throttle rules. Aggressive admission, readmission and insertion settings did not improve the overall result.

The selected dedicated-MMV execution and zero intentionally retained CPU-overlap rows were compared across three paired fresh processes. Code decode improved 80.81→85.29 tokens/sec, medium 77.97→81.73, and extraction 73.54→75.61, with essentially unchanged prefill. This is separate from the much larger total improvement over the original tune.

File-backed canonical expert weights avoid a second large pinned host copy. A default-off loader path suppresses eager file prefetch and releases fully copied source pages after completed uploads, while preserving directly mapped CPU weights. A bounded two-slot pinned staging ring then copies selected expert ranges through 2 MiB buffers, with event-fenced reuse and the original CUDA stream ordering. This recovers some prefill throughput without permanently pinning the entire host expert arena. It adds CPU copying; it does not eliminate all memory traffic. Non-temporal stores were tested and were slower.

The original ordinary cache accepts at most 8 tokens/64 routed rows. A 17-token/136-row extension lets ngram-mod verify a 16-token draft without falling out of that cache. Fresh exact editing improved markedly; ordinary prose did not gain simply from enabling n-grams. Actual target-model verification still determines accepted tokens. This runtime mechanism must not be equated with a model architecture's trained n-gram components.

Further tracing found that real API prompts are split at recurrent-state checkpoints: a 46-token request included 12/30/4-token pieces; a 292-token request included a 265-token prefix and 23/4-token tails. Extending ordinary cache coverage to 32 tokens/256 rows handles these already CPU-scheduled tails. A broader 64-token experiment worked, but its additional CUDA offload override was not responsible for the observed gain. The smaller 32-token variant removes that unnecessary scheduling change. Fused cache operations retain their original 64-row masks and eight-token limit; no public provider ABI or CUDA arithmetic kernel changed.

### Recommended build and runtime

Use the **separate, patched original-fork build**, based on `bd1bf025fc55ffa1fcb2ba6d8bb8805f35671d1f`, with the complete six-file patch. The official newer revision was compatible but did not establish an overall win. No extra startup-warmup, non-temporal-copy, CPU arithmetic, or rows64-routing patch is part of this build.

| Setting | Preserved original | Final recommendation |
|---|---|---|
| Prompt batch / micro-batch | 256 / 128 | 4096 / 4096 |
| Decode / batch threads | 12 / 12 | 6 / 12 |
| First layers with CPU expert storage | 20 | 32 |
| GPU expert-cache budget | Automatic, dormant here | 4480 MiB per device |
| Host weights | Large pinned host copy | Canonical read-only file mapping |
| Speculation | None | ngram-mod, maximum 16 / minimum 2 |
| Model / quantization | Existing Qwen3.6 UD-Q4_K_XL | Identical file |
| Context / K and V precision | 90,000 requested; q8_0 / q8_0 | Unchanged; 90,112 effective |
| Flash Attention / GPU layers / slots | On / 99 / 1 | Unchanged |
| Projector | Existing BF16, GPU offload | Unchanged |
| CPU affinity | Original range arguments are ineffective here | No additional OpenMP binding |

The runner overrides inherit the unchanged model, projector, context, KV and other flags from original-launch.json. The packaged fully resolved selected launch records the complete argv and selected environment; the delta alone is not a standalone command.

Exact additional environment:

```text
GGML_CUDA_MOE_CACHE_BUDGET_MB=4480
GGML_CUDA_MOE_CACHE_RESERVE_MB=512
GGML_CUDA_MOE_CACHE_STATS=10000
GGML_CUDA_MOE_CACHE_HOT_USES=1073741824
GGML_CUDA_MOE_STAGING=1
GGML_CUDA_MOE_STAGING_CHUNK_MB=2
GGML_CUDA_MOE_STAGING_SLOTS=2
GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB=512
LLAMA_MMAP_KEEP_CPU_MOE_HOST=0
LLAMA_MMAP_NO_PREFETCH=1
LLAMA_MMAP_RELEASE_COPIED=1
GGML_CUDA_MOE_CACHE_DEDICATED_MMV=1
GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS=0
GGML_CUDA_MOE_CACHE_ROWS32=1
```

The first source change routes eligible selected expert transfers through the bounded staging callback (`ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`). The second adds the matching ordinary cache row capacity (`ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cuda/moe-cache.cu`). The third controls mmap prefetch and safely releases copied source pages (`src/llama-model.cpp`, `src/llama-model-loader.cpp`). All six before/after files and their hashes are in the source manifest.

The measured binaries reuse unchanged original object files read-only and replace these six translation units. This is not a claim that a clean rebuild will produce identical bytes or inherit these results without testing. Exact incremental compiler/link commands remain in the individual build plans. The independent package contains the exact tested project binaries and CUDA userspace dependencies; compatible host NVIDIA driver and standard system libraries remain required. See reproduction details.

### Other experiments and rejected directions

The complete run catalog and raw summaries accompany this report; the following groups explain the decisions rather than cherry-picking a single favorable run.

| Experiment | Measured finding / decision |
|---|---|
| Batch 512/1024/1536/2048/4096/6144/8192, CPU expert placement and cache budgets | Large prefill gains, but static host placement alone hurts decode. Several high-VRAM combinations OOMed and remain recorded. Selected combinations were validated at 90k context. |
| Latest official upstream `ad6c66839af3c5646fba8c6c2e2087a1e4e38948` | Built separately with a compatible CUDA target. Compatible, with useful configuration-dependent prefill and long-context results, but no demonstrated overall advantage over the selected fork cache path. Version age alone did not win. |
| Existing newer `cc83d7b48` build | Used read-only. About 2,020 prefill / 62.6 native decode at 1024 batching; no overall win. |
| Latest fork `407f…` | Inspected Windows-oriented change; not built and not credited with a performance result. |
| CUDA graph optimizer / graphs disabled | Optimizer produced no consistent gain. Disabling graphs reduced decode to about 55.9; rejected. |
| CUDA blocking / yielding scheduling | Blocking about 55.6 decode; yielding about 61.1 without a repeatable advantage. Rejected. No device-wide setting changed. |
| OpenMP waits, thread counts and affinity | Fewer decode threads and selected cache placement helped. Shorter spin waiting reduced CPU usage but slightly hurt cache decode. Passive waiting and disabling OpenMP were slower. Affinity did not establish a gain. |
| FORCE_MMQ / FORCE_CUBLAS | Source showed FORCE_MMQ is bypassed by this existing Ada/Turing MMA dispatch path, so an identical build was not fabricated as a new experiment. Forced cuBLAS was not built; per-expert conversion/launch costs did not provide a convincing lead. |
| Static one-token CPU MMID scheduling | Passed exact/backend gates but repeated real-model tests regressed; rejected. |
| AVX2 Q5 bit extraction, combined CPU patch, GCC15 CPU backend | Extensive exact arithmetic checks passed. Clean repeated inference showed no robust speed gain; rejected. A Q6 sketch was not compiled because its affected payload was small and no supporting bottleneck evidence emerged. |
| Huge-page pinned host allocator | Warm readiness roughly halved in the pinned-host configuration; useful, separately tested allocator. The ultimately selected file-backed path avoids the large pinned duplicate instead. |
| mmap pin/rebind variants | Several variants passed byte checks but hurt decode or memory. Read-only CUDA registration was unsupported with this CUDA runtime/device; private copies and remapping did not produce the final win. Rejected variants remain isolated. |
| Retained/deferred pinned-host loaders | Exact outputs and dedicated lifetime checks passed; modest loading/RSS effects. Deferred pinning's small extra gain was not accepted as a separate improvement. |
| AVX2 streaming stores / glibc low non-temporal threshold | Medium prefill fell from roughly 1,545 to 1,350/1,320 tokens/sec. Byte-correct but slower; rejected. |
| Auxiliary Qwen3.5-0.8B Q8_0 neural draft | Separate auxiliary file, not a replacement for the 3.6 target. Compatible upstream and higher-headroom original configurations completed but decoded only about 30–34 tokens/sec with roughly 40–45% acceptance. Other configurations stalled. Rejected for performance; the exact stall cause was not proven. |
| q4 KV / TurboQuant V | Saved memory but changed the quality metric on a frozen local source corpus; q8/q8 retained. See quality section. |
| Smaller 8/17/31-token prompt micro-batches | Did not solve cold/API latency overall; tool requests worsened. No speculative adaptive-ubatch source patch was accepted from these screens. |
| Generic BOS/EOS extra startup warmup | Separate common.cpp patch built and screened with 8/31-token extra passes; all arithmetic/tool responses passed, but added 180/251 ms of warmup did not improve launch-to-answer enough. Unselected; broader proposed fault/state fixtures were not run. |
| Late batch-thread and physical-core screens | Batch threads 2/4/6/8/12/16/24 tested on CPU32. 12 remained preferable; 2/24 hurt exact editing. Physical GOMP binding hurt decode; core-spread did not establish a gain. |
| Final measured VRAM headroom | Raising cache 4096→4480 MiB reproduced an exact-edit increase from 187.3/187.5 to 200.7/198.5 tokens/sec in bracketed fresh screens. It uses 384 MiB more VRAM than the smaller-cache experiment; full-capacity checks still fit. Small prose differences were not promoted as separate major gains. |
| Wider fused 136-row or 512-row cache | Source feasibility reviewed, not implemented. Mixed paired residency, multiword masks and additional fallback complexity could negate the savings. The ordinary-row extension achieved a measured gain with less scope. |

Upstream context was checked against the [v0.4.0 release](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0), [PR 28846](https://github.com/ggml-org/llama.cpp/pull/28846), [LRU cache work PR 27861](https://github.com/ggml-org/llama.cpp/pull/27861), and [persistent-cache discussion 28248](https://github.com/ggml-org/llama.cpp/discussions/28248). Their results on other hardware are not substituted for measurements here. The BF16 fallback change in the tested newest revision does not target the target GPU's native BF16 capability. The [reported load-mode memory issue 26110](https://github.com/ggml-org/llama.cpp/issues/26110) motivated inspection, not an assumption that the benchmark host suffered the same problem.

### Correctness, memory interpretation, and remaining limits

The final precision remains the original mixed weight quantization and q8_0 K/V. On a frozen 32,760-target local source corpus, repeated q8/q8 perplexity was 5.5577; q4/q4 was 5.6221 and q8/TurboQuant4 V 5.7131. Paired chunk differences were noisy and dominated by one chunk, so these are descriptive corpus results, not a general statistical quality claim. Lower KV precision was not accepted. The model and projector were never re-quantized or overwritten.

Targeted source tests cover CPU routing and exact arithmetic, actual CUDA cached outputs, duplicate expert IDs, broadcast/per-expert inputs, padded layouts, ordinary/fused boundaries, failed dispatch/collect restoration, staging byte identity, source lifetime, and backend destruction with pending DMA. A cache-disabled full-model staging comparison produced byte-identical review, exact-edit and JSON outputs with staging off/on. Dynamic CPU/GPU cache placement can change floating-point rounding and therefore greedy prose; this is not a claim that every cached model output is byte-identical to the original.

The twelve-fact task has a known baseline error: the original commonly answers 11/12 correctly. It must be compared field by field rather than silently treated as a passing 12/12 test. The final report retains both raw answers and scores. Exact JSON, exact file editing, arithmetic and forced function-call checks provide additional bounded checks. No finite suite proves unchanged quality on every possible prompt.

RSS needs care here. The original CUDA pinned-host allocation largely appears as `Pss_Shmem`; it is not all counted under `Anonymous`. File-backed experts instead appear as `Pss_File`. Therefore higher mapped-file RSS does not mean an equally large increase in physical memory consumption: those file pages can already be present in the OS cache. Compare anonymous plus shared-memory PSS, system available RAM, high-water marks and mapped-file PSS together. File-backed weights are reclaimable under memory pressure and do not guarantee the residency of pinned weights.

The selected 4480 MiB profile passed an **89,700-token input plus 336-token exact answer** within the original 90,112-token allocation. All 25 requested fields were correct. This extended input duplicates only early, non-authoritative snapshot material and preserves the complete original authoritative suffix. It is a capacity/correctness diagnostic, separate from the frozen 81,582-token timing workload. Full tensor validation was enabled for these diagnostic launches and its load timings are not mixed into the ranked load results.

The 512-prefix probability check used the same frozen target continuation and teacher-forced input at every position, reporting the pre-sampling top 32 probabilities. The selected build and original chose the same top token at 507/512 positions (99.02%). Average reference negative-log-likelihood increased by 0.000606 nats, a 1.000606 conditional-perplexity ratio on these 512 positions; all reference tokens appeared in both top 32 lists. Exact extraction agreed at 128/128 probed positions. The small signed reference-NLL mean does not establish uniformly close probability distributions. Mean absolute log-probability differences among common reported top 32 tokens were 0.302 nats for medium, 0.187 for code and 0.313 for extraction; low-probability alternatives can dominate that metric. Bounds derived from the known common-token probabilities and the remaining probability mass put mean full-distribution total-variation distance at approximately 1.05–1.21% for medium, 1.87–2.31% for code and 0.00138–0.00164% for extraction. The largest medium-prefix distance was approximately 26%. Thus top-token agreement and exact extraction were strong, but some prose-prefix distributions changed appreciably. This is not bit-identical inference or a general quality guarantee. Probability-distance bounds record the calculation; no exact full-vocabulary KL is claimed. teacher-summary.json retains per-position results and limits.

Both original and selected builds passed repeated-prefix reuse (9,047 reused tokens), an arithmetic follow-up, disconnecting after 16 streamed events, and a subsequent valid recovery response. Recovery completed in 0.286 s original / 0.332 s selected in this unranked diagnostic. Those are functional recovery measurements, not a claim of a statistically established recovery-speed improvement. See original checks and selected checks.

The remaining limits matter. Very small new-topic requests can still have higher post-readiness latency on the file-backed dynamic-cache path. Generic startup warming, eager mmap prefetch, tiny micro-batches, batch-thread sweeps and affinity did not remove that cost overall. The original pinned-host path also offers stronger host-weight residency under memory pressure. These results establish a substantial real-workload improvement, not universal dominance over every prompt, memory-pressure state or latency submetric.

### Standalone long-context and small-request checks

The independent long-context cohort starts the frozen 81,582-token task as the **first real request**, so no previous answer populates the n-gram table. Two selected launches gave prefill 2,385.74 / 2,368.34 tok/s, decode 57.58 / 57.79 tok/s, and TTFT 34.215 / 34.467 seconds. The interleaved original anchor gave 359.73 prefill, 48.90 decode, and 226.807 seconds TTFT. Selected medians are therefore **2,377.04 prefill, 57.68 decode and 34.341 seconds TTFT**: about 6.61× prefill and 18.0% faster decode. This small standalone cohort supports the direction and separates it from the higher sequential decode figure; it does not pretend to be five independent original anchors. All 25 output fields and API checks passed. Fresh long-context results.

Three paired fresh-server API rounds quantify the small-request exception. These endpoints were non-streaming: the following numbers are **complete response latency, not TTFT**.

| API request/history | Original median | Selected median |
|---|---:|---:|
| First 46-token arithmetic request after readiness | 0.420 s | 0.668 s |
| First 292-token forced tool call, following that arithmetic | 1.432 s | 1.477 s |
| Second identical arithmetic request | 0.393 s | 0.320 s |
| Second identical forced tool call | 1.438 s | 1.061 s |
| Third identical arithmetic request | 0.392 s | 0.311 s |
| Third identical forced tool call | 1.433 s | 1.058 s |

All arithmetic and tool arguments were correct. Warm-cache server readiness in this cohort was 6.892 s original / 1.828 s selected. The first tiny arithmetic response is about 248 ms slower **after readiness**, while startup plus that answer is substantially faster. The approximately 45 ms first-tool difference is small and is not promoted into a broad regression claim. Later repeated answers can benefit from the learned n-gram table even though KV reuse is disabled; they are not fresh-topic measurements. Small-request raw timings.

### Separate cold-file-cache comparison

Two reversed-order pairs began with **0% resident OS file-cache pages** for both the original model and projector, verified before each launch. The method opens those files read-only and requests per-file `POSIX_FADV_DONTNEED`; it does not change their bytes or purge the machine's global caches. Read-only `mmap`/`mincore` records the resulting page-residency snapshot. See Linux's [per-file cache advice](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html) and [residency semantics](https://man7.org/linux/man-pages/man2/mincore.2.html). This does not establish a cold storage-controller cache or cold boot.

The first actual workload is the same 1,015-token exact edit, generating the same correct 938-token answer in all four launches. Medians:

| Cold-file-cache metric | Original | Selected |
|---|---:|---:|
| Server readiness | 46.430 s | 19.455 s |
| First request TTFT after readiness | 2.339 s | 28.035 s |
| Readiness + first TTFT | 48.769 s | 47.490 s |
| Readiness + complete exact-edit response | 62.884 s | 53.207 s |
| First request prefill, including deferred reads | 434.01 tok/s | 36.21 tok/s |
| First request decode | 66.46 tok/s | 164.09 tok/s |

The combined numbers add readiness and request timing, excluding the small harness setup gap. **Do not interpret the 27-second readiness saving as a 27-second first-answer saving.** Most of it shifts disk I/O into the first request. Startup plus first token improves only about 1.28 seconds in this small cohort; the full edit completes about 9.68 seconds sooner. The following medium request is faster, but still slower than the fully warm ranked candidate runs, indicating remaining deferred work. All exact-edit, arithmetic and forced-tool checks passed. Cold metrics, raw cache-state and launch summary.

### Reproduction, evidence, and preservation

For a future model, start with Reproducing with another model. It distinguishes exact-binary replay, source rebuilds and model-specific retuning, including the current harness’s fixed Qwen paths/tokenization.

All experimental code/builds are under `<EXPERIMENT_ROOT>`; original installation paths were read-only inputs. Build plans use isolated object, dependency, log and temporary paths and reuse unchanged original objects read-only. No original CMake/make/ninja build was invoked. The experimental source changes are default-off and selected only through the explicit experimental runtime/environment.

The exact final confirmation runner and frozen comparison summary retain the reported cohort. The general paired runner can reproduce a new five-pair cohort with the same frozen workloads; select a new prefix so existing evidence cannot be overwritten:

```sh
python <EXPERIMENT_ROOT>/final-compare.py \
  --candidate <EXPERIMENT_ROOT>/final-profile.json \
  --prefix independent-recheck-01 --rounds 5
```

This command performs fresh inference and is provided for reproduction; it was not additionally run after the final confirmation. It requires an idle inference GPU and no compilation or heavy background tests. The selected profile now makes inherited batch threads explicitly 12; the resolved-argv check verifies that this documentation change did not alter any effective launch argument.

The evidence includes raw run summaries, CSV metrics, timing exclusions, frozen prompts, original launch, protected-file baseline, and per-run directories with exact commands, responses, logs, telemetry and library mappings. The chronological investigation log preserves intermediate findings, corrections and abandoned approaches. Prepared-but-unrun scripts are not counted as successful experiments.

The investigation catalog contains **276 recorded server runs** (266 completed, 10 failed exploratory variants) and 1,517 request records, including diagnostic probes. Those counts are an inventory, not 276 independent repetitions of the final comparison. Rejected failures, instrumentation, workload caps and cache conditions remain explicit. The selected final confirmation, fresh-long, tiny-request, cold-file-cache and full-capacity diagnostics completed successfully.

The standalone package is bundle-v1. Its manifest records the twelve byte-verified project/CUDA userspace files. The live bundle smoke check verified every selected library mapping, absence of original build-path mappings, a single CPU/CUDA backend file, correct arithmetic, forced tool arguments and the exact edit. The owned server stopped with exit code zero. Packaging did not change binary bytes and its functional smoke is not another ranked speed result.

Inspect or launch the standalone selected profile:

```sh
<EXPERIMENT_ROOT>/selected-rows32-bundle/bundle-v1/start-recommended --dry-run
<EXPERIMENT_ROOT>/selected-rows32-bundle/bundle-v1/start-recommended
```
