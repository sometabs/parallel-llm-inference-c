## StrasGPT: Parallel LLM Inference in C

<p align="center">
  <img src="assets/llama_math-info.png" width="300" height="300" alt="Cute Llama">
</p>

Llama 3.x inference in plain C, parallelized with **OpenMP** and **MPI**. Given a
prompt, it generates the text that continues it.

The sequential implementation is Cédric Bastoul's
[StrasGPT](https://github.com/Ced/StrasGPT), itself building on Andrej
Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and James Delancey's
[llama3.c](https://github.com/jameswdelancey/llama3.c). The parallelization is my
work, done for the Parallel Programming course at Université de Strasbourg.

## What was parallelized

`source/transformer.c` and `source/strasgpt.c`:

- **MPI tensor parallelism**, Megatron-LM style. Each rank loads only its slice of
  the attention heads and the FFN, so weights are never replicated. Two
  `MPI_Allreduce` per layer recombine the row-parallel matmuls before the residual
  adds. Rank 0 samples the next token and broadcasts it.
- **OpenMP** across the matmul, attention and normalization loops.

The head counts and `hidden_dim` must divide by the rank count, so Llama 3.2 1B
runs on 1, 2, 4 or 8 ranks.

## Results

Course VM, 16 cores. 8 prompt tokens, 16 generated.

OpenMP alone on Llama 3.2 1B. Decode scales 5.0x to 16 threads, then degrades once
oversubscribed:

| Threads | 1 | 4 | 8 | 16 | 18 |
|---|---|---|---|---|---|
| Prefill (tok/s) | 9.3 | 27.4 | 46.8 | 61.1 | 34.2 |
| Decode (tok/s) | 6.8 | 18.3 | 30.4 | 34.1 | 13.7 |

MPI, resident memory per rank. This is the point of tensor parallelism: it is what
lets a model larger than one machine's RAM run at all.

| Ranks | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama 3.2 1B | 2.37 GB | 1.46 GB | 1.00 GB | 0.77 GB |
| Llama 3.2 3B | 6.10 GB | 3.46 GB | 2.14 GB | 1.48 GB |

Greedy decoding (`--temp 0`) produces byte-identical output at 1, 2 and 4 ranks,
which is the correctness check for the sharding.

When comparing MPI against OpenMP, pass `--bind-to none`. Otherwise `mpirun` pins a
single-rank job to one core and its threads have nowhere to run.

## Build

```bash
make            # sequential
make parallel   # OpenMP + MPI, needs mpicc
```

`make parallel` does not clean first, so run `make clean` when switching targets or
it will silently reuse objects built with the wrong flags.

Other targets: `make debug` (Valgrind), `make asan` (address sanitizer).

## Get the weights

Any HuggingFace Llama 3.x or Mistral checkpoint works, loaded straight from
safetensors:

```bash
pip install 'huggingface_hub[cli]'
huggingface-cli download unsloth/Llama-3.2-1B --local-dir ../model_zoo/Llama-3.2-1B
huggingface-cli download unsloth/Llama-3.2-3B --local-dir ../model_zoo/Llama-3.2-3B
huggingface-cli download unsloth/Mistral-Nemo-Base-2407 --local-dir ../model_zoo/Mistral-Nemo-Base-2407
```

These are the `unsloth` mirrors, which need no account. The official `meta-llama`
and `mistralai` repositories hold the same weights but are gated, so they need an
approved access request and `huggingface-cli login` first.

The commands below assume the model resolves as `../model_zoo/Llama-3.2-1B` from
the repository root.


## Run

```bash
./strasgpt -m ../model_zoo/Llama-3.2-1B -p "Once upon a time there were three" -n 17
```

```bash
mpirun -np 4 ./strasgpt -m ../model_zoo/Llama-3.2-1B \
  -p "Once upon a time there were three" -n 100 -t 4
```

`scripts/run.sh "your prompt"` wraps the second form, with `RANKS`, `THREADS`,
`STEPS` and `GREEDY` as environment variables. Use `-h` for the full option list.

Sample output:

```
Once upon a time there were three brave knights of the Dark realm. They were
promised that when they conquered the lands and won the hearts of the people,
they would receive golden crowns and a personal holiday every day.

Max memory used (RSS): 0.97 GB
Prompt processing (prefill):    8 tokens in   0.600 s (13.333333 token/s)
Token generation  (decode):    39 tokens in   5.851 s (6.665527 token/s)
```
