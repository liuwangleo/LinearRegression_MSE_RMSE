***************************************
用sciket-learn和pandas学习线性回归
****************************************
1.获取数据 定义问题
  没有数据，当然是无法研究机器学习。这里我们使用UCI大学公开的机器学习数据来跑线性回归
  数据介绍:http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
  下载地址:http://archive.ics.uci.edu/ml/machine-learning-databases/00294/
  这是一个循环发电厂的数据，共有9568个样本数据，每个数据有5列。分别是温度，压力，湿度，压强，输出电力
  我们知道线性回归的目的就是，通过对样本数据的训练，得到一个最优的模型。
  这里温度（AT），压力（V），湿度（AP），压强（RH）是样本特征，PE是对应的样本输出
  PE=θ0+θ1∗AT+θ2∗V+θ3∗AP+θ4∗RH
  而需要学习的，就是θ0,θ1,θ2,θ3,θ4这5个参数。
2.整理数据
  下载后的数据可以发现是一个压缩文件，解压后可以看到里面有一个xlsx文件，我们先用excel把它打开，接着“另存为“”csv格式，保存下来，后面我们就用这个csv来运行线性回归。
   打开这个csv可以发现数据已经整理好，没有非法数据，因此不需要做预处理。但是这些数据并没有归一化，也就是转化为均值0，方差1的格式。也不用我们搞，后面scikit-learn在线性回归时会先帮我们把归一化搞定。
3.用pandas来读取数据
  read_csv读取csv格式的文件
4.准备运行算法的数据
  根据前面的介绍分清楚数据集中那些是：自变量和应变量 就是特征向量和特征值
5.划分训练集和测试集
  把数据集中的X y 一部分作为训练集 测试集
  交叉验证的模块
  需要从sklearn.model_selection导入train_test_split
6.运行sciket-learn的线性模型
  使用线性回归的算法中的最小二乘法来实现模型的拟合
  sk-learn.linear_model 中的LinearRegression
  主要的几个方法 就是：fit predict
7.模型评估
  我们需要对拟合出模型进行好坏程度评估 一般使用均方差或者均方根差
8.交叉验证
  我们可以通过交叉验证来持续优化模型
  sk-learn.model_selection 导入cross_val_predict
9.画图观察结果
    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()