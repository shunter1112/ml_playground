import csv
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import six
import argparse
from PIL import Image
from PIL import ImageOps

test_images = open('test.csv', 'rb')
output = open('answer.csv', 'w')

with open('../trained_model.pkl', 'rb') as model_pickle:
  model = six.moves.cPickle.load(model_pickle)

def forward(x_data):
  x = Variable(x_data)
  h1 = F.relu(model.l1(x))
  h2 = F.relu(model.l2(h1))
  y = F.softmax(model.l3(h2))
  return y

for i,t in enumerate(test_images): 

  if i > 0:
    print '{}\r'.format(i)
    d = map(lambda n:int(n), t.split(","))
    xd = np.asarray([d]).astype(np.float32)/255
    yd = forward(xd).data[0]

    answer = [0,0.]
    for j in range(10):
      if answer[1] < yd[j]:
        answer = [j, yd[j]]
    output.write(str(i))
    output.write(",")
    output.write(str(answer[0]))
    output.write("\n")
  else: 
    output.write("ImageId,Label\n")


# print "-------------------------"
# print "Image should be ", answer[0]