# NVFP4 on Turing

An NVFP4 checkpoint served on a Turing T4, in a Colab notebook.

Turing is compute capability 7.5. It has no FP4 tensor cores, no FP8 tensor cores, and no BF16 —
FP16 is the widest narrow float it has. An NVFP4 checkpoint nonetheless loads and serves there,
because the stored width and the executed width are separate decisions: the weights stay 4-bit in
memory and are unpacked in registers to a width the tensor cores accept.

**This is a proof of concept and not a supported configuration.** It is here so the claim can be
checked, not so it can be deployed.

## What it runs

| | |
|---|---|
| Part | NVIDIA T4, compute capability 7.5, 16 GB |
| Checkpoint | [`nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4) |
| Quantization | `modelopt_mixed` |
| Activation dtype | `float16` — Turing has no BF16 |
| Attention backend | `TRITON_ATTN` |
| Linear backend | `marlin` |
| KV cache element | default, so `float16`; an 8-bit element converts in software here |
| Sequence limit | 512, `max-num-batched-tokens` 512, `max-num-seqs` 1 |
| Memory fraction | 0.88, tensor parallel size 1 |
| Wheels | [`mgschwind/fleetwide-nvfp4-t4-wheelhouse`](https://huggingface.co/datasets/mgschwind/fleetwide-nvfp4-t4-wheelhouse), SHA256-pinned |

Prebuilt wheels are used because building vLLM from source exceeds the free T4 allocation in Colab.
The notebook starts the model download in the background and pins every version it installs, so a
rerun either reproduces or fails loudly.

## What it shows, and what it does not

It shows that the format carries no hardware assumption: a checkpoint stored at 4 bits serves
correctly on an architecture five generations older than the one that executes 4 bits natively. The
sanity check asks for the capital of Austria and gets `Vienna`.

It shows nothing about performance. There is no load sweep, no matched pair, and no throughput
measurement, and the settings — a 512-token limit, one sequence at a time — are chosen to fit a free
Colab session rather than to serve anyone. A T4 is not a serving part and this does not claim it is.

## Reproducing

Open `nvfp4-turing-t4-colab.ipynb` in Colab with a T4 runtime and run the cells in order. Model
download runs in the background while setup proceeds; the offline diagnostic and the `vllm serve`
test are both optional and either one is sufficient to see the checkpoint load and answer.
