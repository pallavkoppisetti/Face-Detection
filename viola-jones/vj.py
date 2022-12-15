import numpy as np

from integral_image import integral_image
from filters import *

class WeakClassifier():
    def __init__(self, feature, threshold, polarity, feature_index) -> None:
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity
        self.feature_index = feature_index

    def predict(self, img):
        iimg = integral_image(img)
        return self._predict_iimg(iimg)

    def _predict_iimg(self, iimg):
        score = self.feature.apply(iimg)
        return 1 if self.polarity * score < self.polarity * self.threshold else 0

    def _predict_ff(self, X_ff):
        score = X_ff[self.feature_index]
        return 1 if self.polarity * score < self.polarity * self.threshold else 0

class VJClassifier:
    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.classifiers = []

    def train(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        weights = np.zeros(len(y))

        pos, neg = 0 , 0

        for i in y:
            if i == 1:
                pos += 1
            else:
                neg += 1

        for i in range(len(y)):
            if y[i] == 1:
                weights[i] = 1 / (2 * pos)
            else:
                weights[i] = 1 / (2 * neg)

        self.features = build_features(X.shape[1], X.shape[2])

        X_ii = np.array([integral_image(x) for x in X])
        self.X_ff = np.array([apply_features(x, self.features) for x in X_ii])

        for i in range(self.feature_count):
            weights = weights / np.sum(weights)
            # train classifier for each feature
            clfs = self._train_clf(weights)
            #select feature with lowest error
            best_clf, best_error, best_acc = self._best_clf(clfs, weights)
            #update weights
            beta = best_error / (1 - best_error)
            weights = np.multiply(weights, np.power(beta, np.subtract(1, best_acc)))
            alpha = np.log(1 / beta)
            print("Best error: ", best_error, "Alpha: ", alpha)
            #add classifier to ensemble
            self.classifiers.append((best_clf, alpha))
        
    def _train_clf(self, weights):

        count_pos, count_neg = 0, 0

        for i in range(len(weights)):
            if self.y[i] == 1:
                count_pos += weights[i]
            else:
                count_neg += weights[i]

        classifiers = []

        for i in range(len(self.features)):

            x_feature = self.X_ff[:, i]

            poss, negs = 0,0
            posw, negw = 0,0

            min_error, best_threshold, best_polarity = float('inf'), 0, 0

            for f, w, label in sorted(zip(x_feature, weights, self.y)):
                error = min(negw + count_pos - posw, posw + count_neg - negw)

                if error < min_error:
                    min_error = error
                    best_threshold = f
                    best_polarity = 1 if poss > negs else -1

                if label == 1:
                    poss += 1
                    posw += w

                else:
                    negs += 1
                    negw += w

            clf = WeakClassifier(self.features[i], best_threshold, best_polarity, i)
            classifiers.append(clf)

        return classifiers

    def _best_clf(self, clfs, weights):
        best_clf = None
        best_error = float('inf')
        best_acc = []

        for clf in clfs:
            error = 0
            acc = []

            for i in range(len(self.X)):
                pred = clf._predict_ff(self.X_ff[i])
                
                mod = abs(pred - self.y[i])
                error += mod * weights[i]
                acc.append(mod)

            if error < best_error:
                best_error = error
                best_clf = clf
                best_acc = acc

        return best_clf, best_error, best_acc

    def predict(self, img):
        iimg = integral_image(img)
        return self._predict_iimg(iimg)

    def _predict_iimg(self, iimg):
        score = 0
        alpha_sum = 0
        for clf, alpha in self.classifiers:
            score += alpha * clf._predict_iimg(iimg)
            alpha_sum += alpha
        return 1 if score >= 0.5 * alpha_sum else 0


    



        
    

