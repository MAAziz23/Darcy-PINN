#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


true_file = "porus_1K"
pred_prefile = "inverse_output_1K_noreg_run"
num_runs = 10


porus_data = pd.read_csv(true_file, sep=r"\s+", header=None)

up_true = porus_data.iloc[:, 2:5].values

def compute_errors(pred_file):

    predicted_data = pd.read_csv(pred_file)

    up_pred = predicted_data[["u_x","u_y","p"]].values

    errors = {}

    names = ["u_x","u_y","p"]

    for j, name in enumerate(names):

        diff = up_pred[:,j] - up_true[:,j]

        errors[f"MAE_{name}"] = np.mean(np.abs(diff))

        errors[f"RelL2_{name}"] = (
            np.linalg.norm(diff)
            / np.linalg.norm(up_true[:,j])
        )

        errors[f"RMSE_{name}"] = np.sqrt(np.mean(diff**2))

    return errors


all_errors = []

for run in range(1, num_runs + 1):

    pred_file = f"{pred_prefile}{run}.csv"

    errors = compute_errors(pred_file)

    errors["run"] = run

    all_errors.append(errors)

errors_df = pd.DataFrame(all_errors)


summary = errors_df.drop(columns=["run"]).agg(["mean","std"]).T

print("\nMean ± SD over 10 runs:\n")

for metric in summary.index:

    mean_val = summary.loc[metric,"mean"]
    std_val  = summary.loc[metric,"std"]

    print(
        f"{metric:15s}: "
        f"{mean_val:.3e} ± {std_val:.3e}"
    )


errors_df.to_csv(
    "all_run_errors_1k_noreg.csv",
    index=False
)

summary.to_csv(
    "summary_mean_std_1k_noreg.csv"
)


# In[ ]:




