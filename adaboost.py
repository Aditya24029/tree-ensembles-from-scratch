import os
import numpy as np
import matplotlib.pyplot as plt


# REPRODUCIBILITY

np.random.seed(42)
os.makedirs("results", exist_ok=True)


# 1. LOAD DATA

data = np.load("data/mnist.npz")

X_train_full = data['x_train']
y_train_full = data['y_train']
X_test_full = data['x_test']
y_test_full = data['y_test']


# 2. FILTER DIGITS 4 & 9

def filter_49(X, y):
    mask = (y == 4) | (y == 9)
    X = X[mask]
    y = y[mask]
    y = np.where(y == 4, -1, 1)
    return X, y

X_train_full, y_train_full = filter_49(X_train_full, y_train_full)
X_test, y_test = filter_49(X_test_full, y_test_full)


# 3. PREPROCESSING

X_train_full = X_train_full.reshape(len(X_train_full), -1) / 255.0
X_test = X_test.reshape(len(X_test), -1) / 255.0


# 4. TRAIN-VAL SPLIT

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


# 5. PCA (TRAIN ONLY)

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

W, mean = compute_pca(X_train, k=5)

X_train = apply_pca(X_train, W, mean)
X_val = apply_pca(X_val, W, mean)
X_test = apply_pca(X_test, W, mean)


# 6. DECISION STUMP

def train_stump(X, y, w):
    n, d = X.shape
    best_err = float('inf')
    best_stump = None

    for j in range(d):
        values = np.sort(np.unique(X[:, j]))
        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2

        # speed optimization
        if len(thresholds) > 1000:
            thresholds = np.random.choice(thresholds, 1000, replace=False)

        for theta in thresholds:
            for polarity in [1, -1]:
                preds = np.ones(n)

                if polarity == 1:
                    preds[X[:, j] <= theta] = 1
                    preds[X[:, j] > theta] = -1
                else:
                    preds[X[:, j] <= theta] = -1
                    preds[X[:, j] > theta] = 1

                err = np.sum(w[preds != y])

                if err < best_err:
                    best_err = err
                    best_stump = (j, theta, polarity)

    return best_stump, best_err

def stump_predict(X, stump):
    j, theta, polarity = stump
    preds = np.ones(X.shape[0])

    if polarity == 1:
        preds[X[:, j] <= theta] = 1
        preds[X[:, j] > theta] = -1
    else:
        preds[X[:, j] <= theta] = -1
        preds[X[:, j] > theta] = 1

    return preds


# 7. ADABOOST TRAINING

T = 300
n = X_train.shape[0]

w = np.ones(n) / n
stumps = []
alphas = []
val_acc = []

print("Training AdaBoost...\n")

for t in range(T):
    stump, err = train_stump(X_train, y_train, w)

    err = max(err, 1e-10)
    alpha = 0.5 * np.log((1 - err) / err)

    preds = stump_predict(X_train, stump)

    w *= np.exp(-alpha * y_train * preds)
    w /= np.sum(w)

    stumps.append(stump)
    alphas.append(alpha)

    # validation prediction
    val_pred = np.zeros(len(X_val))
    for i in range(len(stumps)):
        val_pred += alphas[i] * stump_predict(X_val, stumps[i])

    val_pred = np.sign(val_pred)
    acc = np.mean(val_pred == y_val)
    val_acc.append(acc)

    print(f"Tree {t+1:3d} | Val Accuracy: {acc:.4f}")


# 8. PLOT

plt.figure()
plt.plot(range(1, T+1), val_acc)
plt.xlabel("Number of Trees")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs Number of Trees")
plt.grid()
plt.savefig("results/adaboost_val_accuracy.png", dpi=150)
plt.close()


# 9. BEST ITERATION

best_T = np.argmax(val_acc) + 1
best_val_acc = val_acc[best_T - 1]


# 10. TEST ACCURACY

test_pred = np.zeros(len(X_test))

for i in range(best_T):
    test_pred += alphas[i] * stump_predict(X_test, stumps[i])

test_pred = np.sign(test_pred)
test_acc = np.mean(test_pred == y_test)


# FINAL PRINT

print("\nFINAL RESULTS")
print(f"Best iteration (T*): {best_T}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

