# Studies

Each study contains training recipes, evaluation commands, and model code for
its ablation. Use the study entry points to register custom model families
before loading recipes or checkpoints.

| Study | Paper question | Training runs |
| --- | --- | ---: |
| [`prefill_convergence`](prefill_convergence/README.md) | Figure 5 exact-AR control and time to convergence | 1 control |
| [`rank`](rank/README.md) | Figure 7b rank, static routing, and depth-causal connectivity | 9 rank/baseline arms and 1 depth-causal arm |
| [`schedules`](schedules/README.md) | Figure 7a iteration schedules | 24 cells × 2 seeds |
| [`shared_mixture`](shared_mixture/README.md) | One source mixture projected into 16 KV pairs | shared arm and matched k16 control |

Study outputs belong in `outputs/studies/<study>/<arm>/`, which is ignored by
Git. Activate the installed Python environment before submitting jobs from the
repository root. Supply account, partition, and GPU type through `sbatch` options;
see [the Slurm guide](../slurm/README.md).

Install the optional CUDA evaluation dependencies listed in
[reproduction tools](../docs/reproduction.md) before running a study evaluator.
