from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy import ndimage
# 加载Iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 数据分析和处理
# 1. 查看数据集的统计摘要
print("Data statistics summary:")
print("Mean:", X.mean(axis=0))
print("Standard Deviation:", X.std(axis=0))
print("Minimum:", X.min(axis=0))
print("Maximum:", X.max(axis=0))

# 2. 数据可视化
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Data Visualization")
plt.savefig("data_show.png")

plt.clf()


# 数据去噪和清晰化
# 1. 噪声去除
X = ndimage.median_filter(X, size=3)  # 使用中值滤波去除噪声
# 2. 数据清晰化
X = ndimage.gaussian_filter(X, sigma=1)  # 使用高斯滤波进行数据清晰化
# 将数据集划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42)

# 初始化决策树分类器,设置最大深度实现修剪枝叶
clf = DecisionTreeClassifier(max_depth=3)

# 在训练集上训练决策树模型
clf.fit(X_train, y_train)

# 在验证集上进行预测
y_pred = clf.predict(X_val)

# 计算Micro-F1分数
micro_f1 = f1_score(y_val, y_pred, average='micro')

# 计算Macro-F1分数
macro_f1 = f1_score(y_val, y_pred, average='macro')

# 打印Micro-F1和Macro-F1分数
print("Micro-F1 score: ", micro_f1)
print("Macro-F1 score: ", macro_f1)

# 可视化决策树
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.savefig("result.png")
plt.show()
