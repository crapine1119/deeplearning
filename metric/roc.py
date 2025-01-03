import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

digits = datasets.load_digits()

train_x, test_x, train_y, test_y = train_test_split(
    digits.data, digits.target, train_size=0.5, shuffle=True, stratify=digits.target
)

# intended low performance model for ROC
model = DecisionTreeClassifier(max_features=4)
model.fit(train_x, train_y)

pred = model.predict(test_x)
print("F1 score")
print(f1_score(pred, test_y, average="macro"))

pred_bin = pred / 9

noramlize_factor = np.random.rand(*pred_bin.shape) / 10
pred_bin = (pred_bin + noramlize_factor).clip(0, 1)
test_y_bin = (test_y / 9).round()
##
kde1 = stats.gaussian_kde(pred_bin[test_y_bin == 0])
kde2 = stats.gaussian_kde(pred_bin[test_y_bin == 1])
x = np.arange(0, 1, 0.01)
plt.fill_between(x, kde1(np.arange(0, 1, 0.01)), alpha=0.5, label="0")
plt.fill_between(x, kde2(np.arange(0, 1, 0.01)), alpha=0.5, label="1")
plt.legend()
plt.grid()
##
plt.figure()
tprs, fprs, pres = [], [], []
for threshod in np.arange(0, 1.02, 1e-2):
    pred_for_thr = ((pred_bin - threshod) >= 0).astype(int)

    pre = (test_y_bin[pred_for_thr == 1] == 1).mean()  # precision
    tpr = (pred_for_thr[test_y_bin == 1] == 1).mean()  # recall
    fpr = (pred_for_thr[test_y_bin == 0] == 1).mean()
    tprs.append(tpr)
    fprs.append(fpr)
    pres.append(pre)

plt.plot(fprs, tprs, "k.-", label="ROC")
plt.bar(fprs, pres, width=0.01, label="precision")
plt.grid()
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
# plt.xlim(0, 1)
plt.ylim(0, 1)
