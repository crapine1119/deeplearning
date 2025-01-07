import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

digits = datasets.load_digits()
plt.imshow(digits.images[0], cmap="gray")


train_x, test_x, train_y, test_y = train_test_split(
    digits.data, digits.target, train_size=0.8, shuffle=True, stratify=digits.target
)


def fit_and_pred(model, x, y, test_x, test_y):
    model.fit(x, y)
    pred = model.predict(test_x)

    print(f"Model: {model}")
    print(f"F1: {f1_score(pred, test_y, average="macro")}")


##
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
train_x_stand = scaler.fit_transform(train_x)
test_x_stand = scaler.transform(test_x)

fit_and_pred(DecisionTreeClassifier(), train_x, train_y, test_x, test_y)
fit_and_pred(DecisionTreeClassifier(), train_x_stand, train_y, test_x_stand, test_y)
fit_and_pred(RandomForestClassifier(), train_x, train_y, test_x, test_y)
fit_and_pred(RandomForestClassifier(), train_x_stand, train_y, test_x_stand, test_y)
fit_and_pred(SVC(), train_x, train_y, test_x, test_y)
fit_and_pred(SVC(), train_x_stand, train_y, test_x_stand, test_y)
