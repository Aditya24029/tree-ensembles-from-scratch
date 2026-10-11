import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs("results", exist_ok=True)


# LOAD DATA

train = pd.read_csv('data/fashion-mnist_train.csv')
test = pd.read_csv('data/fashion-mnist_test.csv')

X_train = train.iloc[:, 1:].values / 255.0
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values


# FILTER CLASSES 0,1,2

mask_tr = np.isin(y_train, [0,1,2])
mask_te = np.isin(y_test, [0,1,2])

X_train = X_train[mask_tr]
y_train = y_train[mask_tr]

X_test = X_test[mask_te]
y_test = y_test[mask_te]


# PCA (p = 10)

def compute_pca(X, p):
    mean = np.mean(X, axis=0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:p].T, mean

def apply_pca(X, eigvecs, mean):
    return (X - mean) @ eigvecs

eigvecs, mean = compute_pca(X_train, 10)
Xtr = apply_pca(X_train, eigvecs, mean)
Xte = apply_pca(X_test, eigvecs, mean)


# BEST STUMP (FAST VERSION)

def best_stump(X, y):
    best_f, best_t = None, None
    best_ssr = float('inf')
    best_ml, best_mr = None, None

    for f in range(X.shape[1]):
        x_vals = X[:, f]
        sorted_idx = np.argsort(x_vals)
        x_sorted = x_vals[sorted_idx]

        n = len(x_sorted)

        # sample thresholds (important for speed)
        if n > 200:
            idx_sample = np.linspace(0, n-2, 100, dtype=int)
        else:
            idx_sample = np.arange(n-1)

        thresholds = (x_sorted[idx_sample] + x_sorted[idx_sample+1]) / 2

        for t in thresholds:
            left_mask = x_vals <= t
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left = y[left_mask]
            right = y[right_mask]

            ml = np.mean(left)
            mr = np.mean(right)

            ssr = np.sum((left - ml)**2) + np.sum((right - mr)**2)

            if ssr < best_ssr:
                best_ssr = ssr
                best_f = f
                best_t = t
                best_ml = ml
                best_mr = mr

    return best_f, best_t, best_ml, best_mr


# PREDICTION

def stump_predict(X, f, t, ml, mr):
    preds = np.zeros(len(X))
    preds[X[:, f] <= t] = ml
    preds[X[:, f] > t] = mr
    return preds

def mse(y, yhat):
    return np.mean((y - yhat)**2)


# SINGLE STUMP

f_main, t_main, ml_main, mr_main = best_stump(Xtr, y_train)

pred = stump_predict(Xte, f_main, t_main, ml_main, mr_main)

print("\nSINGLE STUMP")
print("Test MSE:", mse(y_test, pred))


# BAGGING

def bootstrap(X, y):
    n = len(X)
    idx = np.random.choice(n, n, replace=True)
    oob = np.setdiff1d(np.arange(n), idx)
    return X[idx], y[idx], oob

models = []
oob_errors = []

for _ in range(5):
    Xb, yb, oob = bootstrap(Xtr, y_train)

    f, t, ml, mr = best_stump(Xb, yb)
    models.append((f, t, ml, mr))

    if len(oob) > 0:
        pred_oob = stump_predict(Xtr[oob], f, t, ml, mr)
        oob_errors.append(mse(y_train[oob], pred_oob))

print("\nBAGGING")
print("Average OOB MSE:", np.mean(oob_errors))

def bagging_predict(models, X):
    preds = np.array([stump_predict(X, *m) for m in models])
    return np.mean(preds, axis=0)

pred_bag = bagging_predict(models, Xte)

print("Test MSE (Bagging):", mse(y_test, pred_bag))


# PLOT (CORRECT)

plt.figure()

f_plot = f_main

idx = np.argsort(Xte[:, f_plot])

plt.plot(Xte[idx, f_plot], y_test[idx], label='True')
plt.plot(Xte[idx, f_plot], pred[idx], label='Stump')
plt.plot(Xte[idx, f_plot], pred_bag[idx], label='Bagging')

plt.legend()
plt.title("Regression Stump vs Bagging")
plt.xlabel("Feature value")
plt.ylabel("Output")
plt.grid()
plt.savefig("results/stump_vs_bagging.png", dpi=150)

plt.show()
