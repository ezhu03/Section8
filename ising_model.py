import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ---------------------------
# Part 1: Exact Sampling for L=4
# ---------------------------

def compute_energy(config, J=1, B=0):
    """
    Compute the energy of a 2D Ising configuration with periodic boundary conditions.
    Only count right and down neighbors to avoid double counting.
    """
    L = config.shape[0]
    energy = 0.0
    for i in range(L):
        for j in range(L):
            S = config[i, j]
            # Neighbors: right and down (with periodic boundaries)
            right = config[i, (j+1) % L]
            down  = config[(i+1) % L, j]
            energy += -J * S * (right + down)
            energy += -B * S  # external field term
    return energy

def exact_sampling(L=4, J=1, B=0, T=1):
    beta = 1.0 / T
    N = L * L
    configs = []
    energies = []
    probs = []
    
    # Enumerate all possible configurations (2^(L*L) total)
    for bits in product([-1, 1], repeat=N):
        config = np.array(bits).reshape((L, L))
        energy = compute_energy(config, J, B)
        weight = np.exp(-beta * energy)
        configs.append(config)
        energies.append(energy)
        probs.append(weight)
    
    probs = np.array(probs)
    Z = probs.sum()  # partition function
    pdf = probs / Z  # exact probability distribution

    return configs, energies, pdf

# Run exact sampling for L=4 and visualize one sample and the histogram of probabilities.
configs, energies, pdf = exact_sampling(L=4, J=1, B=0, T=1)

# For example, pick the configuration with the highest probability:
max_idx = np.argmax(pdf)
config_max = configs[max_idx]

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.title("Most Probable 4x4 Configuration")
plt.imshow(config_max, cmap='bwr', interpolation='nearest')
plt.colorbar(label='Spin')
plt.subplot(1,2,2)
plt.title("Probability Distribution")
plt.hist(pdf, bins=20, edgecolor='black')
plt.xlabel("Probability")
plt.tight_layout()
plt.show()
plt.savefig('prob_dist.png')

# ---------------------------
# Part 2: Gibbs Sampler Implementation
# ---------------------------

def gibbs_update(config, i, j, J=1, B=0, beta=1.0):
    """
    Update the spin at site (i, j) using the Gibbs sampling rule.
    Uses periodic boundary conditions.
    """
    L = config.shape[0]
    # Sum over the four neighbors
    neighbor_sum = (config[(i+1)%L, j] + config[(i-1)%L, j] +
                    config[i, (j+1)%L] + config[i, (j-1)%L])
    # Compute probability for S(i,j)=+1
    p_up = 1.0 / (1 + np.exp(-2 * beta * (J * neighbor_sum + B)))
    config[i, j] = 1 if np.random.rand() < p_up else -1
    return config

def run_gibbs_sampler(L=10, num_sweeps=1000, burn_in=200, J=1, B=0, T=1):
    beta = 1.0 / T
    # Random initial configuration
    config = np.random.choice([-1, 1], size=(L, L))
    samples = []
    magnetizations = []
    energies = []
    
    for sweep in range(num_sweeps):
        for i in range(L):
            for j in range(L):
                config = gibbs_update(config, i, j, J, B, beta)
        # After each full sweep, record observables (post burn-in)
        if sweep >= burn_in:
            samples.append(config.copy())
            magnetizations.append(np.mean(config))
            energies.append(compute_energy(config, J, B))
    
    return samples, magnetizations, energies

# Example: Run Gibbs sampler for L=10 at T=1 and plot magnetization evolution.
samples, mags, energies_chain = run_gibbs_sampler(L=10, num_sweeps=1000, burn_in=200, J=1, B=0, T=1)
plt.figure()
plt.plot(mags)
plt.xlabel("Sweep Number (post burn-in)")
plt.ylabel("Magnetization")
plt.title("Gibbs Sampler Magnetization Trace (L=10, T=1)")
plt.show()
plt.savefig('gibbs_sampler.png')

# ---------------------------
# Part 3: Phase Transition and Magnetization vs Temperature
# ---------------------------
def temperature_sweep(L=25, num_sweeps=1000, burn_in=200, J=1, B=0, T_vals=None):
    if T_vals is None:
        T_vals = np.linspace(1.5, 3.5, 15)
    avg_mags = []
    for T in T_vals:
        _, mags, _ = run_gibbs_sampler(L=L, num_sweeps=num_sweeps, burn_in=burn_in, J=J, B=B, T=T)
        avg_mags.append(np.mean(np.abs(mags)))  # absolute magnetization as order parameter
    return T_vals, avg_mags

# Sweep temperature for a given lattice size and plot the transition.
T_vals, avg_mags = temperature_sweep(L=25, num_sweeps=1500, burn_in=300, J=1, B=0)
plt.figure()
plt.plot(T_vals, avg_mags, 'o-')
plt.xlabel("Temperature T")
plt.ylabel("Average |Magnetization|")
plt.title("2D Ising Model: Magnetization vs Temperature (L=25)")
# Critical temperature line
T_C = 2 / np.log(1 + np.sqrt(2))
plt.axvline(T_C, color='red', linestyle='--', label=f"T_C ≈ {T_C:.2f}")
plt.legend()
plt.show()
plt.savefig("magnetization_vs_temp.png")

# To reproduce the convergence for various lattice sizes, you would run temperature_sweep for L = 10, 17, 25, 32, 40
# and plot the curves on the same graph.

# ---------------------------
# Part 4: Specific Heat and Magnetic Susceptibility
# ---------------------------
def compute_observables(L=25, num_sweeps=1000, burn_in=200, J=1, B=0, T=2.5):
    beta = 1.0 / T
    _, mags, energies = run_gibbs_sampler(L, num_sweeps, burn_in, J, B, T)
    energies = np.array(energies)
    mags = np.array(mags)
    E_mean = np.mean(energies)
    E2_mean = np.mean(energies**2)
    M_mean = np.mean(mags)
    M2_mean = np.mean(mags**2)
    N = L * L
    
    C_v = beta**2 / N * (E2_mean - E_mean**2)
    chi = beta / N * (M2_mean - M_mean**2)
    return C_v, chi

# Example: Calculate Cv and χ for a range of temperatures.
def observables_vs_temperature(L=25, num_sweeps=1000, burn_in=200, T_vals=None):
    if T_vals is None:
        T_vals = np.linspace(1.5, 3.5, 20)
    C_v_vals = []
    chi_vals = []
    for T in T_vals:
        C_v, chi = compute_observables(L, num_sweeps, burn_in, J=1, B=0, T=T)
        C_v_vals.append(C_v)
        chi_vals.append(chi)
    return T_vals, C_v_vals, chi_vals

T_vals, C_v_vals, chi_vals = observables_vs_temperature(L=25, num_sweeps=1500, burn_in=300)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(T_vals, C_v_vals, 'o-')
plt.xlabel("Temperature T")
plt.ylabel("Specific Heat Cᵥ")
plt.title("Specific Heat vs Temperature")
plt.axvline(T_C, color='red', linestyle='--', label=f"T_C ≈ {T_C:.2f}")
plt.legend()

plt.subplot(1,2,2)
plt.plot(T_vals, chi_vals, 'o-')
plt.xlabel("Temperature T")
plt.ylabel("Susceptibility χ")
plt.title("Magnetic Susceptibility vs Temperature")
plt.axvline(T_C, color='red', linestyle='--', label=f"T_C ≈ {T_C:.2f}")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("specific_heat_susceptibility.png")

def magnetization_vs_field(L=25, num_sweeps=1500, burn_in=300, T=2.5, J=1, B_vals=None):
    """
    Sweep over a range of magnetic field values and compute the average magnetization
    for each using the Gibbs sampler.
    """
    if B_vals is None:
        # Define a range of magnetic fields, for example from -1 to 1
        B_vals = np.linspace(-1, 1, 21)
    avg_mags = []
    for B in B_vals:
        # Run the Gibbs sampler for the current magnetic field value
        _, mags, _ = run_gibbs_sampler(L=L, num_sweeps=num_sweeps, burn_in=burn_in, J=J, B=B, T=T)
        avg_mags.append(np.mean(mags))
    return B_vals, avg_mags

# Run the sweep for a fixed temperature T=2.5 (adjust as needed) and lattice size L=25
B_vals, avg_mags_field = magnetization_vs_field(L=25, num_sweeps=1500, burn_in=300, T=2.5, J=1)

# Plot the magnetization versus magnetic field
plt.figure()
plt.plot(B_vals, avg_mags_field, 'o-', color='purple')
plt.xlabel("Magnetic Field B")
plt.ylabel("Average Magnetization")
plt.title("Magnetization vs Magnetic Field (T=2.5, L=25)")
plt.grid(True)
plt.show()
plt.savefig("magnetization_vs_field.png")
