from integral_image import integral_image
from filters import *
import numpy as np

class WeakClassifier():
    def __init__(self, feature, threshold, polarity, ft_num) -> None:
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity
        self.ft_num = ft_num

    def predict(self, img):
        iimg = integral_image(img)
        return self._predict_iimg(iimg)

    def _predict_iimg(self, iimg):
        score = self.feature.apply(iimg)
        return 1 if self.polarity * score < self.polarity * self.threshold else 0

    def _predict_ff(self, X_ff):
        score = X_ff[self.ft_num]
        return 1 if self.polarity * score < self.polarity * self.threshold else 0 

class RapidObjectDetector:
    def __init__(self, layers):
        self.layer_count = layers
        self.layers = []
        self.X_ff = []
        self.y = []
        self.features = []

    def train(self, X, y) :
        h = X.shape[1]
        w = X.shape[2]

        self.X = X
        # Create integral image
        X_iimg = [integral_image(x) for x in X]
        X_ii = np.array(X_iimg)

        self.features = build_features(w, h)
        X__ff = np.array([apply_features(x, self.features) for x in X_ii])
        self.X_ff = np.array(X__ff)
        self.y = y

        # Initialise AdaBoost weights
        positives = np.count_nonzero(y)
        negatives = len(y) - positives
        
        weights = np.zeros(len(y))

        for i in range(len(y)):
            if y[i] == 1:
                weights[i] = 1 / (2 * positives)
            else:
                weights[i] = 1 / (2 * negatives)

        print(positives, negatives)

        pos , neg = [],[]
        for i, y in enumerate(y):
            if y == 1:
                pos.append(self.X_ff[i])
            else:
                neg.append(self.X_ff[i])

        for i in range(self.layer_count):
            if len(neg) == 0:
                print("No negatives left")
            self._train_layer(i+1, pos + neg, [1 for i in range(len(pos))] + [0 for i in range(len(neg))])
            tp = []
            fp = []
            for sample in neg:
                if self._predict_layer_ff(sample, self.layers[i]) == 1:
                    fp.append(sample)
            
            for sample in pos:
                if self._predict_layer_ff(sample, self.layers[i]) == 1:
                    tp.append(sample)

            pos = tp
            neg = fp

    def _train_weak(self, X_ff, y, weights, features):
        
        pos, neg = 0,0
        
        for i, label in enumerate(y):
            if label == 1:
                pos += weights[i]
            else:
                neg += weights[i]

        clfs = []

        for i, feature in enumerate(features):
            curr_feature_scores = []
            for j in range(len(X_ff)):
                curr_feature_scores.append((X_ff[j][i], y[j], weights[j]))

            min_error = float('inf')
            best_threshold = 0
            best_polarity = 0
            best_feature = i

            posw, negw = 0,0
            poss, negs = 0,0

            curr_feature_scores.sort()
            for x, label, w in curr_feature_scores:
                error = min(negw + pos - posw, posw + neg - negw)
                if error < min_error:
                    min_error = error
                    best_threshold = x
                    best_polarity = 1 if poss > negs else -1

                if label == 1:
                    posw += w
                    poss += 1
                else:
                    negw += w
                    negs += 1
            
            clfs.append((best_feature, best_threshold, best_polarity))

        return clfs

    def _best_feature(self,X_ff, y, clfs, weights):
        best_clf, best_error, best_acc = 0, float('inf'), 0

        for clf in clfs:
            err, acc = 0, []
            assert(len(X_ff) != 0)

            for j,x in enumerate(X_ff):
                chk = abs((1 if clf[2]*x[clf[0]] < clf[2]*clf[1] else 0) - y[j])
                acc.append(abs(chk))
                err += weights[j] * chk

            avg_error = err/len(weights)

            if avg_error < best_error:
                best_clf, best_error, best_acc = clf, avg_error, acc

        print(best_error)
        return WeakClassifier(self.features[best_clf[0]], best_clf[1], best_clf[2], best_clf[0]), best_error, best_acc

    def _train_layer(self, k, X_t, Y_t): 
        cnt_pos = 0
        cnt_neg = 0

        layer = []

        for y in Y_t:
            if y == 1:
                cnt_pos += 1
            else:
                cnt_neg += 1

        weights = np.zeros(len(Y_t))
        
        for i, y in enumerate(Y_t):
            if y == 1:
                weights[i] = 1 / (2 * cnt_pos)
            else:
                weights[i] = 1 / (2 * cnt_neg)

        for i in range(k):
            weights = weights / np.sum(weights)
            clfs = self._train_weak(X_t, Y_t, weights, self.features)
            best_clf, best_error, best_acc = self._best_feature(X_t, Y_t, clfs, weights)
            beta = best_error / (1 - best_error)
            weights = np.multiply(weights, np.power(beta, np.subtract(1, best_acc)))
            layer.append((best_clf, np.log(1/beta)))

        self.layers.append(layer)

    def _predict_layer(self, x, layer):
        x = integral_image(x)
        sum_betas = 0
        score = 0
        for clf, beta in layer:
            sum_betas += beta
            score += beta * clf.predict(x)

        return 1 if score >= 0.5*sum_betas else 0


    def _predict_layer_ff(self, x_ff, layer):
        sum_betas = 0
        score = 0
        for clf, beta in layer:
            sum_betas += beta
            score += beta * clf._predict_ff(x_ff)

        return 1 if score >= 0.5*sum_betas else 0

    def predict(self, x):
        for layer in self.layers:
            if self._predict_layer(x, layer) == 0:
                return 0
        
        return 1

    def faster_predict(self, x):
        h_image = x.shape[0]
        w_image = x.shape[1]

        x_indices = np.ndarray(range(1, 20))
        y_indices = np.ndarray(range(1, 20))

        while x_indices[-1] < w_image and y_indices[-1] < h_image:
            image_window = x[y_indices[0]:y_indices[-1], x_indices[0]:x_indices[-1]]
            if self.predict(image_window) == 1:
                print("Found face")
                return True
            x_indices += 1
            y_indices += 1
            

        