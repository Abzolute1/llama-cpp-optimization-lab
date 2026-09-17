# llama.cpp Optimization Lab

Source patches and experimental methods for improving local mixture-of-experts inference: memory loading, pinned transfers, expert caching, masked attention and speculative decoding.

**Start with [START-HERE.md](START-HERE.md).** It explains the contents, prerequisites, build procedure, validation performed and reproduction limits.

- [Performance report](PERFORMANCE-REPORT.md): measured results and tradeoffs.
- [Reproduction guide](REPRODUCTION-GUIDE.md): experimental method.
- [Pinned source revisions](SOURCE-REVISIONS.json): public upstream inputs.

Both selected patches passed application and resulting-source hash checks. The portable tools passed basic checks. The clean CUDA build route has not yet been validated end to end; historical benchmark scores have not been reproduced from this distribution. Synthetic workloads replace private original inputs.

No model weights or compiled binaries are included. Obtain the matching models and compatible toolchains separately. Existing upstream license notices are retained in `licenses/`; no additional blanket license is asserted by this packaging step.
