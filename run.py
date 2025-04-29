""" Basic stochastic PMD """
import time
import os
import argparse

import numpy as np

import wbmdp 

def policy_update(pi, psi, eta, q=1.0):
    """ Closed-form solution with PMD subproblem

    :param pi (np.ndarray): current policy
    :param psi (np.ndarray): current policy's advantage function
    :param eta (float): step size
    :param q (float): Tsallis entropic index (q = 1 > KL divergence)
    :return (np.ndarray): next policy (should be same shape as @pi)
    """
    
    # Apply psi to update the policy
    if q == 1.0: # KL
        updated = pi * np.exp(-eta * psi.T)
    else: # Tsallis
        pass

    # Normalize across actions
    updated /= np.sum(updated, axis=0, keepdims=True)

    return updated

def simulate_agent(env, pi, T=100):
    s = env.init_agent()
    env.print_grid()
    rng = np.random.default_rng()

    for _ in range(T):
        a = rng.choice(pi.shape[0], p=pi[:,s])
        s = env.step(a)
        time.sleep(0.5)
        env.print_grid()

def train(settings):
    env = wbmdp.get_env(settings['env_name'], settings['gamma'], settings['seed'])

    # print formatter
    exp_metadata = ["Iter", "Est f(pi)", "Est f(pi*)", "Est gap"]
    row_format ="{:>5}|{:>10}|{:>10}|{:>10}"
    print("")
    print(row_format.format(*exp_metadata))
    print("-" * (35+len(exp_metadata)-1))

    # initial policy
    pi_t = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions

    agg_psi_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    agg_V_t = np.zeros(env.n_states, dtype=float)

    if settings['advantage'] == 'linear':
        env.init_estimate_advantage_online_linear({
            "linear_learning_rate": "constant",
            "linear_eta0": 0.1,
            "linear_max_iter": 300,
            "linear_alpha": 0.0001
        })

    s_time = time.time()
    for t in range(settings["n_iters"]):
        if settings['advantage'] == 'generative':
            (psi_t, V_t) = env.estimate_advantage_generative(pi_t, settings["N"], settings["T"]) # Generative
        elif settings['advantage'] == 'linear':
            (psi_t, V_t) = env.estimate_advantage_online_linear(pi_t, settings["T"]) # Online Linear
        elif settings['advantage'] == 'mc':
            (psi_t, V_t, visit_len_state_action) = env.estimate_advantage_online_mc(pi_t, settings["T"]*100, 0*((1 - env.gamma) ** 2) / env.n_actions, True) # Online MC
        else:
            print("ERROR: INCORRECT ADVANTAGE FUNCTION")
        adv_gap = np.max(-agg_psi_t, axis=1)/(1.-env.gamma)

        alpha_t = 1./(t+1)
        agg_psi_t = (1.-alpha_t)*agg_psi_t + alpha_t*psi_t
        agg_V_t = (1.-alpha_t)*agg_V_t + alpha_t*V_t

        if ((t+1) <= 100 and (t+1) % 5 == 0) or (t+1) % 25==0:
            print(row_format.format(
                t+1, 
                "%.2e" % np.dot(env.rho, V_t), 
                "%.2e" % np.dot(env.rho, agg_V_t - adv_gap), 
                "%.2e" % (np.dot(env.rho, V_t) - np.dot(env.rho, agg_V_t - adv_gap)), 
            ))

        # eta_t = settings["alpha"]/(t+1)**0.5
        eta_t = settings["alpha"]/(settings["n_iters"])**0.5
        pi_t = policy_update(pi_t, psi_t, eta_t, settings['divergence_shape'])

    print("Total runtime: %.2fs" % (time.time() - s_time))

    (true_psi_t, true_V_t) = env.get_advantage(pi_t)
    adv_gap = np.max(-true_psi_t, axis=1)/(1.-env.gamma)

    print("=== Final performance metric ===")
    print("  f(pi_k):   %.4e\n  f(pi*) lb: %.4e\n  Gap:       %.4e" % (
        np.dot(env.rho, true_V_t), 
        np.dot(env.rho, true_V_t - adv_gap),
        np.dot(env.rho, adv_gap),
    ))
    print("="*40)

    if settings["visual"] and settings['env_name'] == 'gridworld':
        simulate_agent(env, pi_t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--env_name", type=str, choices=["gridworld", "taxi"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--advantage", type=str, default="generative", choices=["generative", "linear", "mc"])
    parser.add_argument("--divergence", type=float, default=1.0)
    parser.add_argument("--tval", type=int, default=50, help="T value for online MC advantage estimation")

    args = parser.parse_args()

    settings = dict({
        "alpha": args.alpha,
        "visual": args.visual,
        "N": 1,
        "T": args.tval,
        "gamma": 0.9,
        "env_name": args.env_name,
        "n_iters": args.n_iters,
        "seed": args.seed,
        "advantage": args.advantage,
        "divergence_shape": args.divergence
    })

    train(settings)
