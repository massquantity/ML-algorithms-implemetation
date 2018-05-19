import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import zero_one_loss
import time


class AdaBoost(object):
    def __init__(self, M, clf, learning_rate=1.0, method="discrete", tol=None, weight_trimming=None):
        self.M = M
        self.clf = clf
        self.learning_rate = learning_rate
        self.method = method
        self.tol = tol
        self.weight_trimming = weight_trimming

    def fit(self, X, y):
        # tol为early_stopping的阈值，如果使用early_stopping，则从训练集中分出验证集
        if self.tol is not None:      
            X, X_val, y, y_val = train_test_split(X, y, random_state=2)  
            former_loss = 1
            count = 0
            tol_init = self.tol

        w = np.array([1 / len(X)] * len(X))   # 初始化权重为1/n
        self.clf_total = []
        self.alpha_total = []

        for m in range(self.M):
            classifier = clone(self.clf)
            if self.method == "discrete":
                if m >= 1 and self.weight_trimming is not None:
                    # weight_trimming的实现，先将权重排序，计算累积和，再去除权重过小的样本
                    sort_w = np.sort(w)[::-1]     
                    cum_sum = np.cumsum(sort_w)   
                    percent_w = sort_w[np.where(cum_sum >= self.weight_trimming)][0]   
                    w_fit, X_fit, y_fit = w[w >= percent_w], X[w >= percent_w], y[w >= percent_w]
                    y_pred = classifier.fit(X_fit, y_fit, sample_weight=w_fit).predict(X)

                else:
                    y_pred = classifier.fit(X, y, sample_weight=w).predict(X)
                loss = np.zeros(len(X))
                loss[y_pred != y] = 1
                err = np.sum(w * loss)    # 计算带权误差率
                alpha = 0.5 * np.log((1 - err) / err) * self.learning_rate  # 计算基学习器的系数alpha
                w = (w * np.exp(-y * alpha * y_pred)) / np.sum(w * np.exp(-y * alpha * y_pred))  # 更新权重分布

                self.alpha_total.append(alpha)
                self.clf_total.append(classifier)

            elif self.method == "real":
                if m >= 1 and self.weight_trimming is not None:
                    sort_w = np.sort(w)[::-1]
                    cum_sum = np.cumsum(sort_w)
                    percent_w = sort_w[np.where(cum_sum >= self.weight_trimming)][0]
                    w_fit, X_fit, y_fit = w[w >= percent_w], X[w >= percent_w], y[w >= percent_w]
                    y_pred = classifier.fit(X_fit, y_fit, sample_weight=w_fit).predict_proba(X)[:, 1]

                else:
                    y_pred = classifier.fit(X, y, sample_weight=w).predict_proba(X)[:, 1]  
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                clf = 0.5 * np.log(y_pred / (1 - y_pred)) * self.learning_rate
                w = (w * np.exp(-y * clf)) / np.sum(w * np.exp(-y * clf))

                self.clf_total.append(classifier)

            '''early stopping'''
            if m % 10 == 0 and m > 300 and self.tol is not None:
                if self.method == "discrete":
                    p = np.array([self.alpha_total[m] * self.clf_total[m].predict(X_val) for m in range(m)])
                elif self.method == "real":
                    p = []
                    for m in range(m):
                        ppp = self.clf_total[m].predict_proba(X_val)[:, 1]
                        ppp = np.clip(ppp, 1e-15, 1 - 1e-15)
                        p.append(self.learning_rate * 0.5 * np.log(ppp / (1 - ppp)))
                    p = np.array(p)

                stage_pred = np.sign(p.sum(axis=0))
                later_loss = zero_one_loss(stage_pred, y_val)

                if later_loss > (former_loss + self.tol):
                    count += 1
                    self.tol = self.tol / 2  
                else:
                    count = 0
                    self.tol = tol_init
                if count == 2:
                    self.M = m - 20
                    print("early stopping in round {}, best round is {}, M = {}".format(m, m - 20, self.M))
                    break
                former_loss = later_loss

        return self

    def predict(self, X):
        if self.method == "discrete":
            pred = np.array([self.alpha_total[m] * self.clf_total[m].predict(X) for m in range(self.M)])

        elif self.method == "real":
            pred = []
            for m in range(self.M):
                p = self.clf_total[m].predict_proba(X)[:, 1]
                p = np.clip(p, 1e-15, 1 - 1e-15)
                pred.append(0.5 * np.log(p / (1 - p)))

        return np.sign(np.sum(pred, axis=0))


if __name__ == "__main__":
    X, y = datasets.make_hastie_10_2(n_samples=20000, random_state=1)   # data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    start_time = time.time()
    model_discrete = AdaBoost(M=2000, clf=DecisionTreeClassifier(max_depth=1, random_state=1), learning_rate=1.0, 
                              method="discrete", weight_trimming=None)
    model_discrete.fit(X_train, y_train)
    pred_discrete = model_discrete.predict(X_test)
    acc = np.zeros(pred_discrete.shape)
    acc[np.where(pred_discrete == y_test)] = 1
    accuracy = np.sum(acc) / len(pred_discrete)
    print('Discrete Adaboost accuracy: ', accuracy)
    print('Discrete Adaboost time: ', '{:.2f}'.format(time.time() - start_time),'\n')


    start_time = time.time()
    model_real = AdaBoost(M=2000, clf=DecisionTreeClassifier(max_depth=1, random_state=1), learning_rate=1.0, 
                          method="real", weight_trimming=None)
    model_real.fit(X_train, y_train)
    pred_real = model_real.predict(X_test)
    acc = np.zeros(pred_real.shape)
    acc[np.where(pred_real == y_test)] = 1
    accuracy = np.sum(acc) / len(pred_real)
    print('Real Adaboost accuracy: ', accuracy)  
    print("Real Adaboost time: ", '{:.2f}'.format(time.time() - start_time),'\n')

    start_time = time.time()
    model_discrete_weight = AdaBoost(M=2000, clf=DecisionTreeClassifier(max_depth=1, random_state=1), learning_rate=1.0, 
                                     method="discrete", weight_trimming=0.995)
    model_discrete_weight.fit(X_train, y_train)
    pred_discrete_weight = model_discrete_weight.predict(X_test)
    acc = np.zeros(pred_discrete_weight.shape)
    acc[np.where(pred_discrete_weight == y_test)] = 1
    accuracy = np.sum(acc) / len(pred_discrete_weight)
    print('Discrete Adaboost(weight_trimming 0.995) accuracy: ', accuracy)
    print('Discrete Adaboost(weight_trimming 0.995) time: ', '{:.2f}'.format(time.time() - start_time),'\n')

    start_time = time.time()
    mdoel_real_weight = AdaBoost(M=2000, clf=DecisionTreeClassifier(max_depth=1, random_state=1), learning_rate=1.0, 
                                     method="real", weight_trimming=0.999)
    mdoel_real_weight.fit(X_train, y_train)
    pred_real_weight = mdoel_real_weight.predict(X_test)
    acc = np.zeros(pred_real_weight.shape)
    acc[np.where(pred_real_weight == y_test)] = 1
    accuracy = np.sum(acc) / len(pred_real_weight)
    print('Real Adaboost(weight_trimming 0.999) accuracy: ', accuracy)
    print('Real Adaboost(weight_trimming 0.999) time: ', '{:.2f}'.format(time.time() - start_time),'\n')
    
    start_time = time.time()
    model_discrete = AdaBoost(M=2000, clf=DecisionTreeClassifier(max_depth=1, random_state=1), learning_rate=1.0, 
                              method="discrete", weight_trimming=None, tol=0.0001)
    model_discrete.fit(X_train, y_train)
    pred_discrete = model_discrete.predict(X_test)
    acc = np.zeros(pred_discrete.shape)
    acc[np.where(pred_discrete == y_test)] = 1
    accuracy = np.sum(acc) / len(pred_discrete)
    print('Discrete Adaboost accuracy (early_stopping): ', accuracy)
    print('Discrete Adaboost time (early_stopping): ', '{:.2f}'.format(time.time() - start_time),'\n')

    start_time = time.time()
    model_real = AdaBoost(M=2000, clf=DecisionTreeClassifier(max_depth=1, random_state=1), learning_rate=1.0, 
                          method="real", weight_trimming=None, tol=0.0001)
    model_real.fit(X_train, y_train)
    pred_real = model_real.predict(X_test)
    acc = np.zeros(pred_real.shape)
    acc[np.where(pred_real == y_test)] = 1
    accuracy = np.sum(acc) / len(pred_real)
    print('Real Adaboost accuracy (early_stopping): ', accuracy)  
    print('Discrete Adaboost time (early_stopping): ', '{:.2f}'.format(time.time() - start_time),'\n')