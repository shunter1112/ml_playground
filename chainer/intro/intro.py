import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import matplotlib.pyplot as plt
import six

import data
mnist = data.load_mnist_data()

x_all = mnist['data'].astype(np.float32) / 255
y_all = mnist['target'].astype(np.int32)
x_train, x_test = np.split(x_all, [60000])
y_train, y_test = np.split(y_all, [60000])

## Build Model
model = FunctionSet( 
  l1 = F.Linear(784, 100),
  l2 = F.Linear(100, 100),
  l3 = F.Linear(100, 10)
)
optimizer = optimizers.SGD()
optimizer.setup(model)

def forward(x_data, y_data):
  x = Variable(x_data)
  t = Variable(y_data)
  h1 = F.relu(model.l1(x))
  h2 = F.relu(model.l2(h1))
  y = model.l3(h2)
  return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

accuracy_data = []

batchsize = 100
datasize = 60000  
for epoch in range(40):
  print('epoch %d' % epoch)
  indexes = np.random.permutation(datasize)
  for i in range(0, datasize, batchsize):
    x_batch = x_train[indexes[i : i + batchsize]]
    y_batch = y_train[indexes[i : i + batchsize]]
    optimizer.zero_grads()
    loss, accuracy = forward(x_batch, y_batch)
    loss.backward()
    accuracy_data.append(accuracy.data)
    optimizer.update()

sum_loss, sum_accuracy = 0, 0

plt.plot(accuracy_data, 'k--')
plt.show()
plt.savefig("accuracy.png")

for i in range(0, 10000, batchsize):
  x_batch = x_test[i : i + batchsize]
  y_batch = y_test[i : i + batchsize]
  loss, accuracy = forward(x_batch, y_batch)
  sum_loss      += loss.data * batchsize
  sum_accuracy  += accuracy.data * batchsize

mean_loss     = sum_loss / 10000
mean_accuracy = sum_accuracy / 10000

print('mean_loss %.2f' % mean_loss)
print('mean_accuracy %d' % (mean_accuracy * 100))

if (mean_accuracy * 100) > 90:
  with open('trained_model.pkl', 'wb') as output:
    six.moves.cPickle.dump(model, output, -1)
    print "model has saved, it has enough quality as trained model :)"
