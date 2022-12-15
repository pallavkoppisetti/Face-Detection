from load_image import load_train_data, load_test_data
from detector import RapidDetector

X, y = load_train_data(100)

clf = RapidDetector([1,10,25,25,50])
clf.train(X,y)

# Test training accuracy
ps = 0

for i in range(len(X)):
    pred = clf.predict(X[i])
    if pred == y[i]:
        ps += 1

print("Training accuracy: ", ps/len(X))

# Test test accuracy
X, y = load_test_data(15)

ps = 0

for i in range(len(X)):
    pred = clf.predict(X[i])
    if pred == y[i]:
        ps += 1

print("Test accuracy: ", ps/len(X))


    