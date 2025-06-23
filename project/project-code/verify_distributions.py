import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, uniform, weibull_min, kstest
from main import generate_random_trace

# Set seed for reproducibility
seed = 25

# Simulation parameters
time_end = 4000
lambda_rate = 1.6
alpha = 0.8
mu = 0.8
a2l, a2u = 0.9, 1.1  # Uniform scaling bounds for jitter
subjob_probs = [0.22, 0.28, 0.3, 0.08, 0.07, 0.05]  # Probability of subjob counts

# Generate trace data using the simulation function
generated_arrivals, generated_services = generate_random_trace(
    time_end, lambda_rate, a2l, a2u, subjob_probs, mu, alpha, seed=seed
)

# Extract sample arrays for analysis
interarrivals = np.diff(generated_arrivals)  # Interarrival time samples
subjob_counts = np.array([len(s) for s in generated_services])  # Number of subjobs per job
service_samples = np.concatenate(generated_services)  # All service times

# === Isolated validation: exponential distribution ===
exp_samples = interarrivals / np.mean(interarrivals) * (1 / lambda_rate)  # Normalize samples
plt.hist(exp_samples, bins=50, density=True, alpha=0.6, label='Exponential Samples')
x = np.linspace(0, np.max(exp_samples), 300)
plt.plot(x, expon(scale=1/lambda_rate).pdf(x), 'r--', label='Theoretical Exponential')
plt.title('Isolated Exponential Distribution (λ=1.6)')
plt.xlabel('Sample Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.savefig('output/exponential_isolated.png')
plt.clf()

# === Isolated validation: uniform jitter factor ===
uniform_factors = interarrivals / (1 / lambda_rate)  # Extract uniform scaling factors
uniform_factors = uniform_factors[(uniform_factors >= a2l) & (uniform_factors <= a2u)]
plt.hist(uniform_factors, bins=50, density=True, alpha=0.6, label='Uniform Samples')
x = np.linspace(a2l, a2u, 300)
plt.plot(x, uniform(loc=a2l, scale=a2u - a2l).pdf(x), 'r--', label='Theoretical Uniform')
plt.title('Uniform Jitter Factor Distribution (0.9 to 1.1)')
plt.xlabel('Sample Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.savefig('output/uniform_isolated.png')
plt.clf()

# === Combined interarrival time distribution (exponential × uniform) ===
plt.hist(interarrivals, bins=50, density=True, alpha=0.7, label='Samples')
plt.title('Interarrival Time Distribution (Exponential × Uniform)')
plt.xlabel('Interarrival Time')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.savefig('output/interarrival_distribution.png')
plt.clf()

# === Sub-job count distribution ===
unique, counts = np.unique(subjob_counts, return_counts=True)
plt.bar(unique, counts / sum(counts), tick_label=unique)
plt.title('Sub-job Count Distribution')
plt.xlabel('Number of Sub-jobs')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.savefig('output/subjob_count_distribution.png')
plt.clf()

# === Service time distribution (Weibull) ===
plt.hist(service_samples, bins=50, density=True, alpha=0.6, label='Samples')
x = np.linspace(1e-6, np.max(service_samples), 300)
plt.plot(x, weibull_min.pdf(x * mu, alpha) * mu, 'r--', label='Theoretical PDF')
plt.title('Service Time Distribution (Weibull α=0.8 / μ=0.8)')
plt.xlabel('Service Time')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.savefig('output/service_time_distribution.png')
plt.clf()

# === Kolmogorov–Smirnov (KS) tests ===
print("\n[KS Test Results]")
# Note: Approximate fit assuming average uniform factor
k1 = kstest(interarrivals, expon(scale=(1/lambda_rate)*(a2u+a2l)/2).cdf)
print(f"P-value for exponential interarrival samples = {k1.pvalue:.4f}")

# KS test is not appropriate for discrete subjob count
print("P-value for subjob count distribution = N/A (discrete distribution, KS not applicable)")

# KS test for Weibull service times
k3 = kstest(service_samples * mu, weibull_min(alpha).cdf)
print(f"P-value for Weibull service time samples = {k3.pvalue:.4f}\n")
