""" Basic stochastic PMD """
import time
import os
import argparse

import numpy as np

import wbmdp 

def policy_update(pi, psi, eta):
    """ Closed-form solution with PMD subproblem

    :param pi (np.ndarray): current policy
    :param psi (np.ndarray): current policy's advantage function
    :param eta (float): step size
    :return (np.ndarray): next policy (should be same shape as @pi)
    """
    
    # Apply psi to update the policy
    updated = pi * np.exp(-eta * psi.T)

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

    s_time = time.time()
    for t in range(settings["n_iters"]):
        (psi_t, V_t) = env.estimate_advantage_generative(pi_t, settings["N"], settings["T"])
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
        pi_t = policy_update(pi_t, psi_t, eta_t) 

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
    args = parser.parse_args()

    settings = dict({
        "alpha": args.alpha,
        "visual": args.visual,
        "N": 1,
        "T": 50,
        "gamma": 0.9,
        "env_name": args.env_name,
        "n_iters": args.n_iters,
        "seed": args.seed,
    })

    train(settings)
