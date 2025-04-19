# SPMD-gridworld

To train a model:

```
python run.py --env_name gridworld --alpha 1 --n_iters 200 --advantage linear --divergence 1.5
```

`divergence = 1`: KL
`divergence > 1`: Tsallis

To output a visualization of the model in action:

```
python run.py --env_name gridworld --alpha 1 --n_iters 200 --visual
```