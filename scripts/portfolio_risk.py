import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_expected_loss(pd_prob, lgd_pred, EAD):
    return np.mean(pd_prob * lgd_pred * EAD)

def stress_test(pd_prob, lgd_pred, EAD, X_test, n_mc=10000):
    stress_pd = np.clip(pd_prob * (1 + 0.5*(X_test["previous_defaults"] > 1)), 0, 1)
    portfolio_loss = []
    stress_pd_array = stress_pd.to_numpy()
    for _ in range(n_mc):
        sim_default = np.random.binomial(1, stress_pd_array)
        portfolio_loss.append(np.sum(sim_default * lgd_pred * EAD))
    portfolio_loss = np.array(portfolio_loss)
    stress_loss = portfolio_loss.mean()
    portfolio_var = np.percentile(portfolio_loss, 99)
    return portfolio_loss, stress_loss, portfolio_var

def bootstrap_ci(pd_prob, lgd_pred, EAD, n_boot=2000, ci=[2.5,97.5]):
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(len(EAD), len(EAD), replace=True)
        boot_means.append(np.mean(pd_prob[idx] * lgd_pred[idx] * EAD[idx]))
    ci_low, ci_high = np.percentile(boot_means, ci)
    return boot_means, ci_low, ci_high

def plot_histogram(data, save_path, title, xlabel, ylabel):
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()