import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import zero_one_loss
import time


def clock(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        print("training time: ", time.time() - start_time)

    return wrapper


class AdaBoost(object):
    def __init__(self, M, clf, learning_rate=1.0, method="discrete", tol=None, weight_trimming=None):
        self.M = M
        self.clf = clf
        self.learning_rate = learning_rate
        self.method = method
        self.tol = tol
        self.weight_trimming = weight_trimming

    # @clock
    def fit(self, X, y):
        if self.tol is not None:
            X, X_val, y, y_val = train_test_split(X, y, random_state=2)

        w = np.array([1 / len(X)] * len(X))
        self.clf_total = []
        self.alpha_total = []
        former_loss = 1
        count = 0
        tol_init = self.tol

        for m in range(self.M):
            classifier = clone(self.clf)
            if self.method == "discrete":
                if m >= 1 and self.weight_trimming is not None:
                    sort_w = np.sort(w)[::-1]
                    cum_sum = np.cumsum(sort_w)
                    percent_w = sort_w[np.where(cum_sum >= self.weight_trimming)][0]   # 0.999
                    w_fit, X_fit, y_fit = w[w >= percent_w], X[w >= percent_w], y[w >= percent_w]
                    y_pred = classifier.fit(X_fit, y_fit, sample_weight=w_fit).predict(X)
                #    if m % 100 == 0:
                #        print("round {}: {}".format(m, len(w_fit)))
                else:
                    y_pred = classifier.fit(X, y, sample_weight=w).predict(X)
                loss = np.zeros(len(X))
                loss[y_pred != y] = 1
                err = np.sum(w * loss)
                alpha = 0.5 * np.log((1 - err) / err) * self.learning_rate
                w = (w * np.exp(-y * alpha * y_pred)) / np.sum(w * np.exp(-y * alpha * y_pred))

                '''
                percent_w = np.sort(w)[::-1][int(len(w)*0.99)]
                if m >= 300:
                    w_fit, X_fit, y_fit = w[w >= percent_w], X[w >= percent_w], y[w >= percent_w]
                '''
                self.alpha_total.append(alpha)
                self.clf_total.append(classifier)

            elif self.method == "real":
                if m >= 1 and self.weight_trimming is not None:
                    sort_w = np.sort(w)[::-1]
                    cum_sum = np.cumsum(sort_w)
                    percent_w = sort_w[np.where(cum_sum >= self.weight_trimming)][0]
                    w_fit, X_fit, y_fit = w[w >= percent_w], X[w >= percent_w], y[w >= percent_w]
                    y_pred = classifier.fit(X_fit, y_fit, sample_weight=w_fit).predict_proba(X)[:, 1]
                #    if m % 100 == 0:
                #        print(len(X_fit))
                else:
                    y_pred = classifier.fit(X, y, sample_weight=w).predict_proba(X)[:, 1]  ###### X_fit, w_fit
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
                # print("round {}".format(m), zero_one_loss(stage_pred,y_val))
                later_loss = zero_one_loss(stage_pred, y_val)
                if later_loss > (former_loss + self.tol):
                    count += 1
                    self.tol = self.tol / 2  # self.tol = 0.0000001
                #    print(self.tol)
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

        # pred = np.array([0.5 * np.log(self.clf_total[m].predict_proba(X)[:,1] / (1-self.clf_total[m].predict_proba(X)[:,1])) for m in range(self.M)])

        return np.sign(np.sum(pred, axis=0))

    def stage_predict(self, X):
        pred = None
        if self.method == "discrete":
            for alpha, clf in zip(self.alpha_total, self.clf_total):
                current_pred = alpha * clf.predict(X)

                if pred is None:
                    pred = current_pred
                else:
                    pred += current_pred

                yield np.sign(pred)

        elif self.method == "real":
            for clf in self.clf_total:
                p = clf.predict_proba(X)[:, 1]
                p = np.clip(p, 1e-15, 1 - 1e-15)
                current_pred = 0.5 * np.log(p / (1 - p))

                if pred is None:
                    pred = current_pred
                else:
                    pred += current_pred

                yield np.sign(pred)


if __name__ == "__main__":
    X, y = datasets.make_hastie_10_2(n_samples=200000, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # start_time = time.time()
    model = AdaBoost(M=450, clf=DecisionTreeClassifier(max_depth=1, random_state=1), learning_rate=1.0, method="real",
                     tol=0.01, weight_trimming=0.999)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = np.zeros(pred.shape)
    acc[np.where(pred == y_test)] = 1
    accuracy = np.sum(acc) / len(pred)
    print('score: ', accuracy)  # 0.9656667
    # print("execu time: ", time.time() - start_time)

    '''
    start_time = time.time()
    for pred in model.stage_predict(X_test):
        accuracy = 1 - zero_one_loss(y_test, pred)
        print('score', accuracy)
    print("execu time: ", time.time() - start_time)
    '''

    # start_time = time.time()
    base_model = DecisionTreeClassifier(max_depth=1,random_state=1)
    model = AdaBoostRegressor(n_estimators=450, learning_rate=1.0, base_estimator=base_model)
    model.fit(X_train, y_train)
    print("sklearn training time: ", time.time() - start_time)
    pred = model.predict(X_test)
    acc = np.zeros(pred.shape)
    acc[np.where(pred == y_test)] = 1
    accuracy = np.sum(acc) / len(pred)
    print('score_sklearn: ', accuracy)
    # print("execu time: ", time.time() - start_time)

'''
if resample and self.method == "discrete":
    sample = np.random.choice(len(X), len(X), replace=True, p=w)
    y_pred = classifier.fit(X[sample], y[sample]).predict(X)
    loss = np.zeros(len(X))
    loss[y_pred != y] = 1
    err = np.sum(w * loss)
    alpha = 0.5 * np.log((1 - err) / err)
    w = (w * np.exp(-y * alpha * y_pred)) / np.sum(w * np.exp(-y * alpha * y_pred))
    self.alpha_total.append(alpha)
    self.clf_total.append(classifier)

elif resample and self.method == "real":
    sample = np.random.choice(len(X), len(X), replace=True, p=w)
    try:
        y_pred = classifier.fit(X[sample], y[sample]).predict_proba(X)[:, 1]
    except Exception:
        print(m)

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    clf = 0.5 * np.log(y_pred / (1 - y_pred))
    w = (w * np.exp(-y * y_pred)) / np.sum(w * np.exp(-y * y_pred))
    self.clf_total.append(classifier)
    
def clock2(weight):

    def clock(func):

        def wrapper(*args, **kwargs):

            if weight:
                start_time = time.time()
                func(*args, **kwargs)
                print("{} - elapsed time: {}".format("weight_trimming",time.time() - start_time))

            else:
                start_time = time.time()
                func(*args)
                print("{} - elapsed time: {}".format("no weight_trimming", time.time() - start_time))

        return wrapper

    return clock
'''