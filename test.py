# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from planar_utils import *
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model

# 1. 确定各层神经元个数；
# 2. 初始化参数；
# 3. 前向传播，求神经网络的输出值；
# 4. 计算cost；
# 5. 后向传播，求各参数的偏导数值；
# 6. 更新各参数值（使用偏导数、学习效率alpha），完成一次迭代；
# 7. 达到迭代次数，确定神经网络的最终参数；
# 8. 使用参数修正的神经网络预测样本；

def layer_size(X, Y):
    n_x = X.shape[0]# input layer
    n_y = Y.shape[0]
    return n_x, n_y

def initial_para(n_x, n_h, n_y):
    np.random.seed(1)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    para = {"w1":w1, "w2":w2, "b1":b1, "b2":b2}
    return para

def propagation_forward(X, para):
    w1 = para["w1"]
    b1 = para["b1"]
    w2 = para["w2"]
    b2 = para["b2"]
    z1 = np.dot(w1, X) + b1
    A1 = np.tanh(z1)
    z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(z2)

    assert A2.shape == (1, X.shape[1])
    cache = {"z1":z1, "z2":z2, "A1":A1, "A2":A2}
    return A2, cache

def cost_fun(A2, Y):
    m = Y.shape[1]
    cost = -1.0/m * np.sum(Y * np.log(A2) + (1-Y) * np.log(1-A2))
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    return cost

def propagation_back(para, cache, X, Y):
    w1 = para["w1"]
    b1 = para["b1"]
    w2 = para["w2"]
    b2 = para["b2"]
    m = Y.shape[1]
    A2 = cache["A2"]
    A1 = cache["A1"]
    dz2 = A2-Y
    dw2 = 1/m * np.dot(dz2, A1.T)
    assert dw2.shape == w2.shape
    db2 = 1/m * np.sum(dz2,axis=1,keepdims=True)    # 按行相加，并且保持其二维特性
    assert db2.shape == b2.shape
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(A1, 2))
    dw1 = 1/m * np.dot(dz1, X.T)
    assert dw1.shape == w1.shape
    db1 = 1/m * np.sum(dz1,axis=1,keepdims=True)
    assert db1.shape == b1.shape
    grads = {"dw1":dw1, "db1":db1, "dw2":dw2, "db2":db2}
    return grads

def update_para(para, grads, learning_rate):
    w1 = para["w1"]
    b1 = para["b1"]
    w2 = para["w2"]
    b2 = para["b2"]
    ### END CODE HERE ###

    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dw1 = grads["dw1"]
    db1 = grads["db1"]
    dw2 = grads["dw2"]
    db2 = grads["db2"]

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    para = {"w1": w1, "w2": w2, "b1": b1, "b2": b2}
    return para

def predict(para, X):
    # X的维度是（nx, m）的，Z = model(np.c_[xx.ravel(), yy.ravel()])
    # 注意这两处的ｘ的维度需要一致，不一样需要转置
    A2, cache = propagation_forward(X, para)
    pre = np.around(A2)
    return pre

def model(X, Y, iteration_num, n_h = 4):
    learning_rate = 1.2
    n_x, n_y = layer_size(X, Y)
    para = initial_para(n_x, n_h, n_y)
    for i in range(iteration_num):
        A2, cache = propagation_forward(X, para)
        cost = cost_fun(A2, Y)
        grads = propagation_back(para, cache, X, Y)
        para = update_para(para, grads, learning_rate)
        if i % 1000 == 0:
            print("%d Times, Cost is %f" % (i, cost))
    return para

def main():
    # X, Y = load_planar_dataset()
    # # (400,2).T   (400,1).T 前200个是0,后200个是1
    # # plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), s=40, cmap=plt.cm.Spectral);
    # # plt.show()
    #
    # # 逻辑回归做2分类
    # # m = X.shape[1]
    # # clf = sklearn.linear_model.LogisticRegressionCV()
    # # clf.fit(X.T, Y.T)
    # # # plot_decision_boundary这里的函数用的x，y是（nx，m）型的，与draw——decision不一样，该处只需要用（m，nx）
    # # plot_decision_boundary(lambda x:clf.predict(x), X, Y)
    # # plt.show()
    # # LR_pre = clf.predict(X.T).reshape(Y.shape)
    # # assert LR_pre.shape == Y.shape
    # # accuracy = 100 - 100 * np.mean(np.abs(LR_pre-Y))
    # # print("Training data accuracy:%f" % accuracy , "%")
    # # Training data accuracy:47.000000 %
    #
    # # 用浅层的神经网络
    # # parameters = model(X, Y, iteration_num=10000)
    # # predictions = predict(parameters, X)
    # # # 画出分界
    # # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # # plt.title("Decision Boundary for hidden layer size " + str(4))
    # # plt.show()
    # # accuracy = 100 - 100 * np.mean(np.abs(predictions - Y))
    # # print("Training data accuracy:%f" % accuracy , "%")
    #
    # # 改变神经元的个数
    # plt.figure(figsize=(16, 32))
    # layer = [1, 2, 3, 4, 5, 10, 20]
    # for i, n_h in enumerate(layer):
    #     plt.subplot(5, 2, i+1)
    #     plt.title("Hidden_size is %d" % n_h)
    #     parameters = model(X, Y, iteration_num=10000, n_h=n_h)
    #     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    #     plt.title("Decision Boundary for hidden layer size " + str(n_h))
    #
    #     predictions = predict(parameters, X)
    #     accuracy = 100 - 100 * np.mean(np.abs(predictions - Y))
    #     print("Training data accuracy:%f" % accuracy , "%")
    # plt.show()
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    X, Y = blobs
    X, Y = X.T, Y.reshape(1, Y.shape[0]) % 2
    # 取余的原因是只进行二分类ｘ（２，　２００） Y (1, 200)
    # plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), s=40, cmap=plt.cm.Spectral);
    # plt.show()
    n_h = 10
    parameters = model(X, Y, iteration_num=10000, n_h=n_h)
    predictions = predict(parameters, X)
    # 画出分界
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(n_h))
    plt.show()
    accuracy = 100 - 100 * np.mean(np.abs(predictions - Y))
    print("Training data accuracy:%f" % accuracy , "%")
    pass

if __name__ == '__main__':
    main()