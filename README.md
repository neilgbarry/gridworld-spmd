# SPMD-gridworld

To train a model:

```
python run.py --env_name gridworld --alpha 1 --n_iters 200 --advantage linear --divergence 1
```

`divergence = 1`: KL

To output a visualization of the model in action:

```
python run.py --env_name gridworld --alpha 1 --n_iters 200 --visual
```

The `run.py` script includes several input parameters that configure the behavior of the stochastic Policy Mirror Descent algorithm. These parameters are passed through command-line arguments and parsed using Python's `argparse` library. The most essential input is `--alpha`, the learning rate controlling the step size for policy updates. The `--env\_name` parameter selects between predefined environments (e.g., `"gridworld"` or `"taxi"`), while `--seed` sets the random seed for reproducibility. The number of training iterations is controlled via `--n\_iters`. The algorithm supports different advantage estimation methods through the `--advantage` argument, which accepts `"generative"`, `"linear"`, or `"mc"` (Monte Carlo). Additionally, `--tval` sets the trajectory length `T` used in Monte Carlo estimation. The `--visual` flag triggers environment visualization for `gridworld`, providing a step-by-step display of agent behavior under the learned policy.

Main experiments
- `spmd_bigger_gridworld.ipynb`: exploring larger grid in gridworld
- `spmd_subtask.ipynb`: adding a subtask and modifying reward structure of the problem
- `spmd_function_approximation.ipynb`: function approximation and online learning
- `plots.py`: some of the generative experiments and codes for plotting (to be added to the repo)
