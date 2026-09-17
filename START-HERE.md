# Source and benchmark reproduction package

This package contains two selected inference-backend patches, pinned source identities, portable launch profiles, numerical historical results, a CUDA fixture and a new synthetic benchmark harness. It contains no private application source, original application prompts, conversations, configuration backups, compiled binaries or model weights.

## Read in this order

1. `PERFORMANCE-REPORT.md`: measured gains, rejected experiments and limitations.
2. `REPRODUCTION-GUIDE.md`: experimental method and cross-model considerations.
3. This file: the portable source package and its checks.

The historical reports refer to additional evidence that is not included. Numerical summaries here preserve timing values but omit private workload content and generated answers. The new synthetic workloads exercise similar mechanisms; their scores must never be presented as reproductions of the historical numbers.

## What is included

- `patches/qwen35-selected.patch`: the original selected six-file patch, byte-for-byte unchanged. It adds bounded pinned staging, mapped-weight loader controls and wider ordinary expert-cache coverage.
- `patches/flash-next-selected.patch`: a consolidated three-file patch containing registration dependencies, private writable mappings, optional prefault workers and masked-attention skipping. Consolidation includes the earlier registration dependency, not just the final incremental patch.
- `patches/*-source-manifest.json`: pinned base revisions and before/after hashes for every affected file.
- `profiles/`: baseline and candidate launch parameters with explicit model, projector and port placeholders.
- `evidence/`: selected native numerical timing summaries. Output text, prompt text, output hashes, application tests and host inventory are omitted.
- `fixtures/test-maskskip.cpp`: the original synthetic CUDA attention fixture, unchanged. It generates its own tensors; it contains no application data.
- `tools/`: new portable preparation, launch and synthetic benchmark tools.
- `SOURCE-REVISIONS.json`: public source URLs, commits and source archive checksums.
- `licenses/`: original upstream license notices.

## Reproduction boundary

The historical candidates were assembled using compatible existing objects and selected replacement libraries. The build helper provides a **new clean-build route**, which has not been compiled or benchmarked as part of preparing this package. It does not promise byte-identical historical binaries or historical performance. Compatible NVIDIA drivers, CUDA, C/C++ compilers, CMake, Ninja, Git and Python 3 are required on the recipient's machine.

The Flash-Next patch consistently renames custom environment switches to the neutral `LLAMA_EXPERIMENT_` prefix. The launch profile uses those names. No numerical kernel or loader logic was changed by this rename. Existing historical binaries do not recognize the renamed switches: rebuild from this patch. Registration uses a presence check; omit `LLAMA_EXPERIMENT_REGISTER_COW` to disable it. Assigning `0` still enables it. The selected prefault thread count is zero.

Source revisions and model identities are public technical inputs. Hostnames, personal paths, device identifiers and private application internals are not required by the package. Experimental context, cache budgets, thread counts and placement values are retained as parameters, not claims that another machine has the same resources. Check all affinity ranges and memory requirements before launch.

## Obtain sources

For the 35B patch:

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant source-qwen35
git -C source-qwen35 checkout bd1bf025fc55ffa1fcb2ba6d8bb8805f35671d1f
python3 tools/prepare-build.py qwen35 --source source-qwen35 --check-only
```

For Flash-Next, obtain the candidate source archive identified in `SOURCE-REVISIONS.json`, verify its SHA-256, and extract it into a fresh `source-flash` directory. Its pinned revision is `329b6160f513915f1c607dbfae3d5ce864a64a4f`.

```bash
python3 tools/prepare-build.py flash-next --source source-flash --check-only
```

`--source` must name the extracted directory containing `CMakeLists.txt`, not its enclosing download directory. Never apply the patches to a production installation. The helper refuses mismatched affected-file hashes.

## Build

Set `CUDA_ARCH` to the CMake architecture value appropriate for the recipient's GPU, then build in an unused directory:

```bash
python3 tools/prepare-build.py qwen35 --source source-qwen35 --build build-qwen35 --cuda-arch "$CUDA_ARCH" --jobs 4
python3 tools/prepare-build.py flash-next --source source-flash --build build-flash --cuda-arch "$CUDA_ARCH" --jobs 4
```

These commands modify only the explicitly supplied source checkout and build directory. Use separate fresh source trees for controls. `--baseline` builds unmodified source. A same-revision unmodified control isolates patch effects; it is different from comparing against a historical older Flash-Next release. The original Flash-Next source archive is also identified in `SOURCE-REVISIONS.json` for that separate comparison.

Select a CUDA-compatible host compiler through the standard compiler environment or an explicitly reviewed local CMake configuration. Never mix libraries from different builds or incompatible ABI variants. The helper uses native CPU compilation; its output is intended for the build host, not as a universal binary distribution.

## Launch

Obtain the matching model weights separately. For the 35B profile, also supply the matching BF16 projector; the historical experiment included it for memory parity even on text-only tasks. All paths are supplied by the recipient:

```bash
python3 tools/launch.py --profile profiles/qwen35-candidate.json --server build-qwen35/bin/llama-server --model "$MODEL" --projector "$PROJECTOR" --port 18081 --dry-run
python3 tools/launch.py --profile profiles/qwen35-candidate.json --server build-qwen35/bin/llama-server --model "$MODEL" --projector "$PROJECTOR" --port 18081
```

For Flash-Next:

```bash
python3 tools/launch.py --profile profiles/flash-next-candidate.json --server build-flash/bin/llama-server --model "$MODEL" --port 18081
```

The profiles retain measured configuration values rather than automatically fitting an arbitrary GPU. The Flash-Next private-mapping path can commit tens of GiB of RAM; 90,000-token contexts and expert-cache budgets also require substantial memory. Reduce settings only as an explicitly labeled new experiment. Baseline and candidate runs must use consistent model precision, context and sampling.

## Synthetic benchmark

Wait for the explicitly launched server to become ready. The harness only sends requests to a loopback endpoint; it does not launch, kill or reconfigure processes.

```bash
python3 tools/benchmark.py --url http://127.0.0.1:18081 --model benchmark-model --workload edit --output edit-candidate-01.json
```

Use the server's actual model alias if different. Other workloads are `extract` and `prose`. The edit checks exact source bytes including the trailing newline; extraction checks the expected integer; prose requires manual quality review. Output records wall time and server token usage, not streamed TTFT or separately measured decode speed. These are smoke measurements, not a replacement for the historical full benchmark harness.

Restart the server between independent samples so the n-gram table cannot learn previous answers. Alternate baseline/candidate order, use fresh result filenames, and run repeated matched samples. Do not compile, hash model files, run GPU fixtures or perform heavy disk I/O during timed inference. A speed result counts only alongside correctness and explicit cache-state conditions.

## CUDA fixture

`fixtures/test-maskskip.cpp` takes a backend-library directory and an output directory. Compile it against the matching Flash-Next build's ggml headers and libraries. Run it once with unmodified libraries and once with patched libraries, then compare the generated numerical outputs and timings. A passing synthetic fixture does not prove full-model numerical equivalence. This fixture was included unchanged, but was not compiled or executed during packaging.

## Validation performed for this package

- Both patches applied successfully to temporary copies of the pinned affected source files.
- Every resulting affected source file matched its public manifest's after-hash: six files for the 35B patch and three for Flash-Next.
- Portable Python tools passed syntax and command-line checks; preparation, launch substitution and benchmark response handling were exercised without live inference.
- Included JSON parses successfully; sharing files were scanned for personal paths, private project identifiers and common credential patterns.
- No live model inference, clean CUDA rebuild or reproduction of historical benchmark results was performed while packaging.

`SHA256SUMS` records the distributed file contents. Verify with `sha256sum -c SHA256SUMS` from this directory.
