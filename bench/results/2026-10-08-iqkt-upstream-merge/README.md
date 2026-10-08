# IQ_KT: upstream 0.1.40.3 merge on 2080 Ti + 4080 SUPER

The [merge report (中文)](../../../docs/IQ_KT_UPSTREAM_MERGE.zh-CN.md) describes the configuration,
builds, validation, interpretation and deployment. The baseline is `a0b5b3b`; the merged binary was
built at `15743b6`, which contains upstream `d5ea713` and the local IQ_KT/Vulkan work.

`results.json` contains allowlisted measurements exported from the local engine protocol and 0.5 s
telemetry. It includes individual request times, output token SHA-256 hashes, sample counts, mean,
median, range and sample standard deviation. Warm-up rows are marked and excluded from comparisons.
API keys, private server presets, raw prompts and deployment credentials are not included.
`validation.json` records the numerical checks, 260K retrieval, MTP smoke tests, exact-name Intel
selection and five real HTTP requests. It includes the synthetic test answers and engine configuration.
The benchmark binary hashes are in `results.json`; the backed-up and deployed binary hashes are in
`validation.json`. The pre-merge benchmark uses the separate reviewed build of the same baseline source.

The main comparison is old A → new A → old B → new B. Each arm has one warm-up and two rounds of
72-, 3,570- and 12,150-token inputs, with 256 generated tokens for each measured request. All arms
use K8V8, 262,144 context capacity, 20,480 resident KV cells, 2048 prefill chunks, a 19/29 layer split,
full expert residency and MTP off. The same request-triggered GPU wake provider runs after engine
READY. Results show no clear throughput change from the merge.

The optional RAM PLE arm is separate: it has only two samples per input length and was not alternated
against direct PLE. It must not be folded into the old/new comparison. The 260K arm is a retrieval and
capacity check with eight output tokens. The MTP arm uses split 20 and prefill 512 and checks the native
KT sidecar; its throughput is not comparable with the main control.

GPU clocks are sampled over approximate decode intervals inferred from host request start and engine
prefill time. Available RAM statistics distinguish whole-process measurements (including startup)
from each request; in particular, the RAM PLE arm's lowest available RAM occurred during startup,
before mapped expert pages were trimmed from the working set.

Original per-arm files remain under the local `build-kt/upstream-*.{json,log,protocol}` and
`build-kt/upstream-*-telemetry.jsonl`. The report lists the additional numerical and HTTP checks.
