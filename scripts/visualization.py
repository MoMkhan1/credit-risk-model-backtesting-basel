import matplotlib.pyplot as plt
import pandas as pd

def plot_pd_histogram(pd_prob, save_path="figures/pd_histogram.png"):
    plt.figure()
    plt.hist(pd_prob, bins=50)
    plt.title("Predicted Probability of Default")
    plt.xlabel("PD"); plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()

def plot_lgd_scatter(y_true, lgd_pred, save_path="figures/lgd_scatter.png"):
    plt.figure()
    plt.scatter(y_true, lgd_pred, alpha=0.3)
    plt.title("Predicted vs Actual LGD")
    plt.xlabel("Actual LGD"); plt.ylabel("Predicted LGD")
    plt.savefig(save_path)
    plt.close()

def plot_bootstrap_distribution(boot_means, ci_low, ci_high, save_path="figures/bootstrap_ci.png"):
    plt.figure()
    plt.hist(boot_means, bins=50)
    plt.axvline(ci_low, color='r')
    plt.axvline(ci_high, color='r')
    plt.title("Bootstrap Expected Loss Distribution")
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, features, save_path="figures/logreg_feature_importance.png"):
    import pandas as pd
    import matplotlib.pyplot as plt
    importance = pd.Series(model.coef_[0], index=features).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    importance.plot(kind='barh')
    plt.title("Logistic Regression Feature Importance")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()