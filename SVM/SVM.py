import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 生成SVM数据
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [-1] * 20 + [1] * 20

# 训练SVM模型
model = SVC(kernel='linear')
model.fit(X, Y)

# 获取支持向量和超平面参数
support_vectors = model.support_vectors_
w = model.coef_[0]
b = model.intercept_[0]

# 计算超平面的斜率和截距
slope = -w[0] / w[1]
intercept = -b / w[1]

# 输出支持向量
print("Support Vectors:")
for vector in support_vectors:
    print(vector)

# 绘制数据点和超平面
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='r', marker='o', label='Support Vectors')
plt.plot(X[:, 0], slope * X[:, 0] + intercept, 'k-', label='Hyperplane')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Hyperplane')
plt.savefig('result.png')
plt.show()
