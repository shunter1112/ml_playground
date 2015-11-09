import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
from PIL import Image
from PIL import ImageOps
import six


img = Image.open('sample.png')
# img = ImageOps.grayscale(img)
# im = np.asarray(img.resize((4,4))).astype(np.float32)
# np.reshape(im, ())

# img = Image.open(args.img)
img = ImageOps.grayscale(img)
xd = np.asarray(img.resize((28,28))).reshape((1,784)).astype(np.float32) 
xd = xd / 100.
# print xd

import data
mnist = data.load_mnist_data()

x_all = mnist['data'].astype(np.float32) / 255
y_all = mnist['target'].astype(np.int32)
x_train, x_test = np.split(x_all, [60000])
y_train, y_test = np.split(y_all, [60000])

print y_all
# print x_train[0]