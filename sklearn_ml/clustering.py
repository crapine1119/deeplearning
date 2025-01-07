import pandas as pd
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

fp = "/Users/songhak/Downloads/거래데이터 -Onlinesales_info-전처리 - 시트1.csv"

data = pd.read_csv(fp)
data.groupby("대 카테고리").count()
label = data["대 카테고리"]

c2i_high = {c: enum for enum, c in enumerate(label.unique())}
i2c_high = {v: k for k, v in c2i_high.items()}

c2i_low = {c: enum for enum, c in enumerate(data["제품카테고리"].unique())}
i2c_low = {v: k for k, v in c2i_low.items()}

id2i = {c: enum for enum, c in enumerate(data["제품ID"].unique())}
i2id = {v: k for k, v in id2i.items()}

data_x = data.copy()
data_x["대 카테고리"] = data_x["대 카테고리"].apply(lambda x: c2i_high[x])
data_x["제품카테고리"] = data_x["제품카테고리"].apply(lambda x: c2i_low[x])
data_x["제품ID"] = data_x["제품ID"].apply(lambda x: id2i[x])

# target_column = data_x.describe().columns
target_column = data.describe().columns
data_x = data_x[target_column]
data_y = label.apply(lambda x: c2i_high[x])

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=0.8, shuffle=True, stratify=data_y)

scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.fit_transform(test_x)

max_cluster = 10
inertia_result = []
for n_clusters in range(1, max_cluster + 1):
    model = cluster.KMeans(n_clusters=n_clusters)
    model.fit(train_x_scaled)
    inertia = model.inertia_
    inertia_result.append(inertia)

plt.plot(inertia_result, "o-")
plt.grid()
plt.ylabel("inertia")
plt.xlabel("n_cluster")
##
inertia_diff_ths = 20000
best_cluster_index = max_cluster - 1
min_diff = float("inf")
for i in range(max_cluster - 1):
    diff = inertia_result[i] - inertia_result[i + 1]
    if diff < inertia_diff_ths:
        min_diff = min(diff, min_diff)
        best_cluster_index = i
        print(f"best cluster index: {i}")
        break

best_n_cluster = best_cluster_index + 1
model = cluster.KMeans(n_clusters=best_n_cluster)
model.fit(train_x_scaled)

pred_y = model.predict(test_x_scaled)

pred_data = test_x.copy()
pred_data["pred"] = pred_y
pred_data = pred_data.sort_index()

f1_score(pred_y, test_y.values, average="micro")
silhouette_score(test_x, pred_y)
calinski_harabasz_score(test_x, pred_y)
davies_bouldin_score(test_x, pred_y)
##
decomp = PCA(n_components=2)
pca_x = decomp.fit_transform(test_x_scaled)
plt.figure()
for c in range(best_n_cluster):
    x = pca_x[pred_y == c, 0]
    y = pca_x[pred_y == c, 1]
    plt.scatter(x, y, label=c)
##
main_stat = ["count", "mean", "min", "max"]
for c in range(best_n_cluster):
    cluster_index = pred_data[pred_data["pred"] == c].index
    cluster_data = data.loc[cluster_index]
    print(f"cluster {c}")
    freq_result = []
    for col in cluster_data.columns:
        desc = cluster_data[col].describe()
        if "top" in desc:
            freq_result.append(desc[["top", "freq"]])

    print(pd.concat(freq_result, axis=1))
    print(cluster_data.describe().loc[main_stat])
    print("=" * 100)
