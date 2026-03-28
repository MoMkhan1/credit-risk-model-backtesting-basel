import numpy as np
import pandas as pd
import os

def generate_credit_data(n=50000, save_path="data/credit_data.csv", random_state=42):
    np.random.seed(random_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Generate correlated features
    mean = [45, 50000, 700]
    cov = [[100, 2000, 50], [2000, 40000000, 500], [50, 500, 400]]
    age_income_score = np.random.multivariate_normal(mean, cov, n)
    age = np.clip(age_income_score[:,0], 18, 70)
    income = np.clip(age_income_score[:,1], 20000, 200000)
    credit_score = np.clip(age_income_score[:,2], 300, 850)

    # Additional features
    loan_amount = np.random.normal(15000, 5000, n)
    loan_term = np.random.randint(6, 60, n)
    credit_cards = np.random.randint(0, 5, n)
    previous_defaults = np.random.randint(0, 3, n)

    # PD probability (~10–20% defaults)
    logit = -5 + 0.01*(age-45) - 0.00002*(income-50000) - 0.008*(credit_score-700) + 0.6*previous_defaults
    PD = np.random.binomial(1, 1 / (1 + np.exp(-logit)))

    # LGD and EAD
    LGD = np.clip(0.2 + 0.5*(700-credit_score)/400 + 0.1*np.random.rand(n), 0.2, 0.8)
    EAD = np.random.uniform(5000, 20000, n)

    # Create DataFrame
    df = pd.DataFrame({
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_cards": credit_cards,
        "previous_defaults": previous_defaults,
        "credit_score": credit_score,
        "PD": PD,
        "LGD": LGD,
        "EAD": EAD
    })

    df.to_csv(save_path, index=False)
    return df