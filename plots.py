import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

# Settings
n_trials = 10
n_iters = 250
env_name = 'gridworld'
alpha_list = np.logspace(-1, 1, 7)
t_list = np.arange(20, 120, 20)

# Compile regex once
pattern = re.compile(r"\s*(\d+)\|\s*[\deE\+\-\.]+\|\s*[\deE\+\-\.]+\|\s*([\deE\+\-\.]+)")


iters = []
# Run trials for each alpha
all_results = {}
for t in t_list:
	print(f"Running t={t:.3f}")
	trial_gaps = []

	for trial in range(n_trials):
		print(f"  Trial {trial+1}/{n_trials}")

		process = subprocess.Popen(
			['python', 'run.py',
			'--env_name', env_name,
			'--alpha', str(50),
			'--n_iters', str(n_iters),
			'--seed', str(trial),
			'--tval', str(t)],  # use seed = trial
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True
		)

		stdout, stderr = process.communicate()

		if stderr:
			print(f"Trial {trial} errors:", stderr)

		gaps = np.full(n_iters, np.nan)

		for line in stdout.splitlines():
			match = pattern.match(line)
			if match:
				iter_num = int(match.group(1))
				if iter_num <= n_iters:
					gaps[iter_num-1] = float(match.group(2))
		iters = [index+1 for index, x in enumerate(gaps) if not np.isnan(x)]
		gaps = [x for x in gaps if not np.isnan(x)]
		trial_gaps.append(gaps)

	trial_gaps = np.array(trial_gaps)  # (n_trials, n_iters)
	mean_gaps = np.nanmean(trial_gaps, axis=0)
	all_results[t] = mean_gaps

# Plot
plt.figure(figsize=(10, 6))
# iters = np.arange(5, n_iters+1, 5)
for alpha, mean_gaps in sorted(all_results.items()):
	plt.plot(iters, mean_gaps, label=f'T={alpha}', marker='.')

plt.xlabel('Iteration')
plt.ylabel('Gap')
plt.yscale('log')
plt.grid(True)
plt.legend(title="Step Size α", loc='best', fontsize='small')
plt.title(f'Convergence Comparison over {n_trials} Trials per T')
plt.tight_layout()
plt.savefig('convergence_comparison.png', dpi=300)
plt.show()
