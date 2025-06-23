import numpy as np
import matplotlib.pyplot as plt
from main import ServerFarm, generate_random_trace

# Fixed simulation parameters
n = 8  # Number of servers
lambda_rate, a2l, a2u = 1.6, 0.9, 1.1  # Interarrival rate and scaling bounds
subjob_probs = [0.22, 0.28, 0.3, 0.08, 0.07, 0.05]  # Probability distribution of number of sub-jobs
mu, alpha = 0.8, 0.8  # Weibull distribution parameters for service times
time_end = 4000  # Duration of the simulation
runs_per_h = 30  # Number of simulation runs for each threshold h
h_values = list(range(0, 9))  # Different threshold values to test

results = {}  # Dictionary to store mean and CI for each h

print("Running simulations to find best h...\n")
for h in h_values:
    mrt_list = []
    for _ in range(runs_per_h):
        # Generate one set of random arrivals and services
        arrivals, services = generate_random_trace(
            time_end, lambda_rate, a2l, a2u, subjob_probs, mu, alpha
        )
        # Initialize server farm and run simulation
        sf = ServerFarm(n_servers=n, threshold=h)
        sf.simulate(arrivals, services, time_end)

        # Compute response time for each job
        job_response_times = [
            max(departs) - arrivals[job_id]
            for job_id, departs in sf.job_completion.items()
        ]
        mrt = np.mean(job_response_times)
        mrt_list.append(mrt)

    # Calculate statistics: mean MRT, 95% confidence interval
    mean = np.mean(mrt_list)
    std = np.std(mrt_list, ddof=1)
    se = std / np.sqrt(runs_per_h)
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se
    results[h] = (mean, ci_lower, ci_upper)

    print(f"h = {h}: MRT = {mean:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

# Visualization: MRT vs h with error bars for CI
means = [results[h][0] for h in h_values]
lowers = [results[h][0] - results[h][1] for h in h_values]  # Distance from mean to lower bound
uppers = [results[h][2] - results[h][0] for h in h_values]  # Distance from mean to upper bound

plt.errorbar(h_values, means, yerr=[lowers, uppers], fmt='o-', capsize=5, ecolor='red')
plt.xlabel('Threshold h')
plt.ylabel('Mean Response Time (MRT)')
plt.title('MRT vs Threshold h with 95% Confidence Interval')
plt.grid(True)
plt.tight_layout()
plt.savefig("output/mrt_vs_h.png")  # Save the plot
plt.show()
