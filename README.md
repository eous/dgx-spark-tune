# dgx-spark-tune

One-file tuning script for the NVIDIA DGX Spark. Checks and applies inference-focused optimizations, then validates everything is set up correctly.

It was written for the GB10 (Blackwell, aarch64, unified memory) but most of the tuning applies to any Linux GPU box.

## What it does

- Kernel sysctls: BBR congestion control, socket buffers, memory pressure tuning for unified GPU/CPU memory
- Docker: NVIDIA runtime as default, log rotation, container ulimits
- WiFi: disables power save for lower latency
- NVIDIA: persistence mode, CUDA toolkit (installs latest from repo)
- Sleep prevention: masks systemd sleep/suspend/hibernate, configures logind and GNOME
- GPU clock autotune: benchmarks clock speeds with a 1000-iteration matmul kernel, picks the best, persists it as a systemd service

## Usage

```
# check everything, fix interactively
sudo python3 dgx-spark-tune.py

# check only, no changes (good for CI or monitoring)
sudo python3 dgx-spark-tune.py --check

# apply all fixes without prompting
sudo python3 dgx-spark-tune.py --apply

# run the GPU clock autotune (~12 min) and apply the result
sudo python3 dgx-spark-tune.py --autotune --apply
```

## Requirements

- Python 3.10+
- Root access
- NVIDIA GPU with `nvidia-smi`
- `nvcc` (CUDA toolkit) for `--autotune` only

## Example output

```
=== DGX Spark Inference Tuning Validator ===

  Kernel sysctls (network + VM)
  [ OK ] sysctl net.core.default_qdisc           fq
  [ OK ] sysctl net.ipv4.tcp_congestion_control   bbr
  ...

  CUDA toolkit
  [ OK ] cuda-toolkit installed                   13.2
  [ OK ] cuda-toolkit is latest                   13.2

  GPU clock lock (autotuned)
  [ OK ] GPU clock locked at 2600 MHz             2600 MHz, service active

  50 checks: 50 passed

  System Summary
    Host:     flatspark (flatspark.blck.io)
    CPU:      Cortex-A725 (20 cores)
    Memory:   121.7 GB total, 118.6 GB free
    GPU:      NVIDIA GB10
    GPU mem:  unified with system RAM (121.7 GB shared)
    CUDA:     13.2 (driver 580.142)
```

## License

MIT
