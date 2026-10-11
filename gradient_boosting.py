import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs("results", exist_ok=True)


# 1. LOAD DATA

data = np.load("data/mnist.npz")

X_train_full = data['x_train']
y_train_full = data['y_train']
X_test_full = data['x_test']
y_test_full = data['y_test']


# 2. FILTER 4 & 9

def filter_49(X, y):
    mask = (y == 4) | (y == 9)
    X = X[mask]
    y = y[mask]
    y = np.where(y == 4, -1, 1)
    return X, y

X_train_full, y_train_full = filter_49(X_train_full, y_train_full)
X_test, y_test = filter_49(X_test_full, y_test_full)


# 3. PREPROCESS

X_train_full = X_train_full.reshape(len(X_train_full), -1) / 255.0
X_test = X_test.reshape(len(X_test), -1) / 255.0


# 4. SPLIT

def split_train_val(X, y):
    X_train, y_train, X_val, y_val = [], [], [], []

    for cls in [-1, 1]:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)

        val_idx = idx[:1000]
        train_idx = idx[1000:]

        X_val.append(X[val_idx])
        y_val.append(y[val_idx])
        X_train.append(X[train_idx])
        y_train.append(y[train_idx])

    return (np.vstack(X_train), np.hstack(y_train),
            np.vstack(X_val), np.hstack(y_val))

X_train, y_train, X_val, y_val = split_train_val(X_train_full, y_train_full)


# 5. PCA

def compute_pca(X, k=5):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:k]]

    return W, mean

def apply_pca(X, W, mean):
    return np.dot(X - mean, W)

W, mean = compute_pca(X_train, 5)

X_train = apply_pca(X_train, W, mean)
X_val = apply_pca(X_val, W, mean)
X_test = apply_pca(X_test, W, mean)


# 6. STUMP

def train_stump_ssr(X, y):
    n, d = X.shape
    best_loss = float('inf')
    best_stump = None

    for j in range(d):
        values = np.sort(np.unique(X[:, j]))
        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2

        if len(thresholds) > 1000:
            thresholds = np.random.choice(thresholds, 1000, replace=False)

        for theta in thresholds:
            left = X[:, j] <= theta
            right = ~left

            if np.sum(left) == 0 or np.sum(right) == 0:
                continue

            c1 = np.mean(y[left])
            c2 = np.mean(y[right])

            preds = np.zeros(n)
            preds[left] = c1
            preds[right] = c2

            loss = np.sum((y - preds) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_stump = (j, theta, c1, c2)

    return best_stump

def stump_predict(X, stump):
    j, theta, c1, c2 = stump
    preds = np.zeros(X.shape[0])

    left = X[:, j] <= theta
    preds[left] = c1
    preds[~left] = c2

    return preds


# 7. GRADIENT BOOSTING

def gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test, eta, verbose=True):
    T = 300

    F_train = np.zeros(len(X_train))
    F_val = np.zeros(len(X_val))
    F_test = np.zeros(len(X_test))

    stumps = []
    val_mse = []

    for t in range(T):
        residual = y_train - F_train

        stump = train_stump_ssr(X_train, residual)

        F_train += eta * stump_predict(X_train, stump)
        F_val += eta * stump_predict(X_val, stump)

        mse = np.mean((y_val - F_val) ** 2)
        val_mse.append(mse)

        stumps.append(stump)

        if verbose and (t % 10 == 0):
            print(f"Tree {t+1:3d} | Val MSE: {mse:.5f}")

    best_T = np.argmin(val_mse) + 1

    for i in range(best_T):
        F_test += eta * stump_predict(X_test, stumps[i])

    test_mse = np.mean((y_test - F_test) ** 2)

    return val_mse, best_T, test_mse


# 8. MAIN RUN (η = 0.01)

eta_main = 0.01

val_mse_main, best_T_main, test_mse_main = gradient_boosting(
    X_train, y_train, X_val, y_val, X_test, y_test, eta_main, verbose=True
)

# Show main plot
plt.figure()
plt.plot(range(1, 301), val_mse_main)
plt.xlabel("Number of Trees")
plt.ylabel("Validation MSE")
plt.title(f"MSE vs Trees (eta={eta_main})")
plt.grid()
plt.savefig("results/gb_mse_eta_0.01.png", dpi=150)



# 9. RESULTS

print("\n==============================")
print(f"Best Number of Trees: {best_T_main}")
print(f"Best Validation MSE: {min(val_mse_main):.5f}")
print(f"Test MSE: {test_mse_main:.5f}")
print("==============================")


# 10. MULTIPLE ETA

etas = [0.001, 0.01, 0.1, 0.2, 0.5, 1]

print("\nLearning Rate Analysis:")

for eta_val in etas:
    print(f"\nRunning eta={eta_val}...")

    val_mse, best_T, test_mse = gradient_boosting(
        X_train, y_train, X_val, y_val, X_test, y_test, eta_val, verbose=True
    )

    print(f"eta={eta_val:<6} | Best T={best_T:<3} | Test MSE={test_mse:.5f}")

    plt.figure()
    plt.plot(range(1, 301), val_mse)
    plt.xlabel("Number of Trees")
    plt.ylabel("Validation MSE")
    plt.title(f"MSE vs Trees (eta={eta_val})")
    plt.grid()
    plt.savefig(f"results/gb_mse_eta_{eta_val}.png", dpi=150)

plt.show()    
