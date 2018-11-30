import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 导入划分训练测试 交叉验证的模块
from sklearn.model_selection import train_test_split,cross_val_predict
# 线性回归的模块
from sklearn.linear_model import LinearRegression
# 计算均方差的模块
from sklearn import metrics

# 使用read_csv读取csv格式的文件
data = pd.read_csv('.\CCPP\ccpp.csv')
# 读取前5行 如果是读取后5行使用head.tail()
print(data.head())
print(data.shape)
# 特征向量 特征值
X = data[['AT', 'V', 'AP', 'RH']]
print(X.head())
y = data[['PE']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 线性回归算法使用最小二乘法实现拟合
# 创建线性回归的对象
linreg = LinearRegression()
# 进行数据的拟合
linreg.fit(X_train, y_train)
# 拟合出的截距
print(linreg.intercept_)
# 拟合出的斜率 相当于 θ（1-4）参数
print(linreg.coef_)

# 模型的评估
# 模型拟合测试集
y_pre = linreg.predict(X_test)
# 计算均方差
print("############均方差#################")
print("MSE:", metrics.mean_squared_error(y_test, y_pre))
print("############均方根差#######################")
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pre)))

# 交叉验证
pre = cross_val_predict(linreg,X,y,cv=10)
print("#############交叉验证##############")
print(pre)
print("############均方差#################")
print("MSE:", metrics.mean_squared_error(y, pre))
print("############均方根差#######################")
print("RMSE:", np.sqrt(metrics.mean_squared_error(y, pre)))


#画图观察结果
fig,ax = plt.subplots()
ax.scatter(y,pre)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()