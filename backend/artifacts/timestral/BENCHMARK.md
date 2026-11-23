# Timestral PumpWatch Benchmark

Run date: 2026-10-21

The four official Timestral checkpoints were tested locally on CPU with eight
PyTorch threads. InfluxDB credentials were not configured, so accuracy was
measured on one deterministic PumpWatch-shaped synthetic holdout containing
four pumps, eleven target variates, 1,024 context readings and a 64-reading
forecast horizon. Severe phase-current loss, flow loss, under-voltage and
power-factor drop events were injected only after the forecast was produced.

Lower normalized MAE, normalized pinball loss and false-positive rate are
better. Coverage should be close to the nominal 90% interval.

| Model | Parameter memory | Normalized MAE | Pinball | 90% coverage | q01/q99 false positives | Injected-fault recall | 4-pump warm latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5M | 19.87 MB | 0.217463 | 0.145894 | 88.03% | 2.64% | 100% | 1.87 s |
| 28M | 106.13 MB | 0.210017 | 0.142235 | 88.14% | 2.49% | 100% | 5.15 s |
| 123M | 467.71 MB | 0.201279 | 0.134960 | 90.38% | 2.08% | 100% | 14.61 s |
| 209M | 798.46 MB | 0.191673 | 0.129242 | 91.80% | 2.05% | 100% | 24.37 s |

The API-shaped latency probe used one pump, eleven target variates, 350 context
readings and a one-step forecast.

| Model | Median warm latency |
|---|---:|
| 5M | 0.0855 s |
| 28M | 0.4072 s |
| 123M | 1.4402 s |
| 209M | 2.1876 s |

## Recommendation

Use Timestral-123M for the first shadow integration. It had near-target
interval calibration and nearly the same injected-fault false-positive rate as
209M while using about 41% less parameter memory and responding about 34%
faster in the one-pump probe. Keep 28M as the lower-latency deployment option.
Do not make the final production selection until the same rolling-origin
benchmark has been run on real, regularly sampled InfluxDB telemetry.

Reproduce a full benchmark from the `backend` directory:

```powershell
.\.venv\Scripts\python.exe benchmarks\timestral_benchmark.py artifacts\timestral\checkpoints\Timestral-123M
```

Reproduce the API-shaped latency probe:

```powershell
.\.venv\Scripts\python.exe benchmarks\timestral_benchmark.py artifacts\timestral\checkpoints\Timestral-123M --context-length 350 --latency-only
```
