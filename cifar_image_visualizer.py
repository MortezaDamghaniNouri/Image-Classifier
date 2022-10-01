from keras.datasets import cifar10
import matplotlib.pyplot as plt

(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
n = 100
plt.figure(figsize=(20, 10))
for i in range(n):
    plt.imshow(train_X[i])
    plt.show()

