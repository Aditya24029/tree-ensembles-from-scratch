import os
import numpy as np

np.random.seed(42)
os.makedirs("results", exist_ok=True)


# LOAD + PREPROCESS

data = np.load('data/mnist.npz')

X_train = data['x_train']
y_train = data['y_train']
X_test  = data['x_test']
y_test  = data['y_test']

mask_tr = np.isin(y_train, [0,1,2])
mask_te = np.isin(y_test,  [0,1,2])

X_train = X_train[mask_tr].reshape(-1, 784) / 255.0
y_train = y_train[mask_tr]
X_test  = X_test[mask_te].reshape(-1, 784) / 255.0
y_test  = y_test[mask_te]


# PCA (SVD)

def compute_pca(X, p):
    mean = np.mean(X, axis=0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:p].T, mean

def apply_pca(X, W, mean):
    return (X - mean) @ W

W, mean = compute_pca(X_train, 10)
Xtr = apply_pca(X_train, W, mean)
Xte = apply_pca(X_test,  W, mean)


# GINI FUNCTIONS

CLASSES = [0,1,2]

def gini(y):
    if len(y) == 0:
        return 0
    return 1 - sum((np.mean(y == c))**2 for c in CLASSES)

def weighted_gini(l, r):
    n = len(l) + len(r)
    return (len(l)/n)*gini(l) + (len(r)/n)*gini(r)

def majority(y):
    counts = [np.sum(y == c) for c in CLASSES]
    return CLASSES[np.argmax(counts)]


# BEST SPLIT

def best_split(X, y, features=None):
    if features is None:
        features = range(X.shape[1])

    best_f, best_t, best_score = None, None, float('inf')

    for f in features:
        t = np.mean(X[:, f])

        mask = X[:, f] <= t
        yl, yr = y[mask], y[~mask]

        if len(yl) == 0 or len(yr) == 0:
            continue

        score = weighted_gini(yl, yr)

        if score < best_score:
            best_f, best_t, best_score = f, t, score

    return best_f, best_t, best_score


# TREE STRUCTURE

def leaf(y):
    return {'leaf': True, 'label': majority(y)}

def node(f, t, left, right):
    return {'leaf': False, 'feat': f, 'thresh': t,
            'left': left, 'right': right}


# BUILD TREE (3 LEAVES)

def build_tree(X, y, k=None):

    def get_features():
        if k is None:
            return None
        return np.random.choice(X.shape[1], k, replace=False)

    f1, t1, _ = best_split(X, y, get_features())
    if f1 is None:
        return leaf(y)

    mask = X[:, f1] <= t1
    Xl, yl = X[mask], y[mask]
    Xr, yr = X[~mask], y[~mask]

    fl, tl, gl = best_split(Xl, yl, get_features())
    fr, tr, gr = best_split(Xr, yr, get_features())

    n = len(y)

    left_split  = (len(Xl)/n)*gl + (len(Xr)/n)*gini(yr) if fl is not None else np.inf
    right_split = (len(Xl)/n)*gini(yl) + (len(Xr)/n)*gr if fr is not None else np.inf

    if left_split <= right_split and fl is not None:
        m = Xl[:, fl] <= tl
        left_node = node(fl, tl, leaf(yl[m]), leaf(yl[~m]))
        right_node = leaf(yr)
    elif fr is not None:
        m = Xr[:, fr] <= tr
        left_node = leaf(yl)
        right_node = node(fr, tr, leaf(yr[m]), leaf(yr[~m]))
    else:
        left_node = leaf(yl)
        right_node = leaf(yr)

    return node(f1, t1, left_node, right_node)


# PREDICTION

def predict_one(tree, x):
    while not tree['leaf']:
        if x[tree['feat']] <= tree['thresh']:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree['label']

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])

def accuracy(y, yhat):
    return np.mean(y == yhat)

def classwise(y, yhat):
    for c in CLASSES:
        m = (y == c)
        acc = np.mean(yhat[m] == c)
        print(f"Class {c} accuracy: {acc:.4f}")


# SINGLE TREE

tree = build_tree(Xtr, y_train)
pred = predict(tree, Xte)

print("\nSingle Tree Accuracy:", accuracy(y_test, pred))
classwise(y_test, pred)


# BAGGING

N = len(Xtr)

bootstraps = []
oobs = []
trees = []
oob_errs = []

for i in range(5):
    idx = np.random.choice(N, N, replace=True)
    oob = np.setdiff1d(np.arange(N), idx)

    bootstraps.append(idx)
    oobs.append(oob)

    t = build_tree(Xtr[idx], y_train[idx])
    trees.append(t)

    if len(oob) > 0:
        pred_oob = predict(t, Xtr[oob])
        oob_errs.append(1 - accuracy(y_train[oob], pred_oob))

print("\nAvg OOB error (Bagging):", np.mean(oob_errs))

def vote(trees, X):
    preds = np.array([predict(t, X) for t in trees])
    final = []
    for i in range(X.shape[0]):
        counts = [np.sum(preds[:, i] == c) for c in CLASSES]
        final.append(CLASSES[np.argmax(counts)])
    return np.array(final)

pred_bag = vote(trees, Xte)
print("Bagging Accuracy:", accuracy(y_test, pred_bag))
classwise(y_test, pred_bag)


# RANDOM FOREST (FIXED)

k = 3

rf_trees = []
oob_rf = []

for i in range(5):
    idx = bootstraps[i]   # SAME bootstrap
    oob = oobs[i]

    t = build_tree(Xtr[idx], y_train[idx], k)
    rf_trees.append(t)

    if len(oob) > 0:
        pred_oob = predict(t, Xtr[oob])
        oob_rf.append(1 - accuracy(y_train[oob], pred_oob))

print("\nAvg OOB error (RF):", np.mean(oob_rf))

pred_rf = vote(rf_trees, Xte)
print("RF Accuracy:", accuracy(y_test, pred_rf))
classwise(y_test, pred_rf)
