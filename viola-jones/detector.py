import numpy as np
from vj import VJClassifier


class RapidDetector:
    def __init__(self,layer_counts):
        self.layer_counts = layer_counts
        self.layers = []

    def train(self, X, y):
        pos , neg = [], []

        for i in range(len(y)):
            if y[i] == 1:
                pos.append(X[i])
            else:
                neg.append(X[i])

        for feature_count in self.layer_counts:
            if len(neg) == 0:
                break
            print("Training layer with {} features".format(feature_count))
            vj = VJClassifier(feature_count)
            vj.train(np.array(pos+neg), [1]*len(pos)+[0]*len(neg))
            self.layers.append(vj)

            fp = []
            for sample in neg:
                if vj.predict(sample) == 1:
                    fp.append(sample)

            neg = fp

    def predict(self, X):
        for layer in self.layers:
            if layer.predict(X) == 0:
                return 0
        return 1

