import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import six
import argparse
from PIL import Image
from PIL import ImageOps

parser = argparse.ArgumentParser(
    description='A Neural Algorithm of Artistic Style')
parser.add_argument('--img', '-i', default='',
                    help='path of input image')
args = parser.parse_args()

with open('trained_model.pkl', 'rb') as model_pickle:
  model = six.moves.cPickle.load(model_pickle)

def forward(x_data):
  x = Variable(x_data)
  h1 = F.relu(model.l1(x))
  h2 = F.relu(model.l2(h1))
  y = F.softmax(model.l3(h2))
  return y

img = Image.open(args.img)
img = ImageOps.invert(img)
img = ImageOps.grayscale(img)
xd = np.asarray(img.resize((28,28))).reshape((1,784)).astype(np.float32) / 100.
print xd
yd = forward(xd).data[0]

answer = [0,0.]
for i in range(10):
  print "Probability of ", i, " : ", int(yd[i] * 10000)/100.0, "%" 
  if answer[1] < yd[i]:
    answer = [i, yd[i]]

print "-------------------------"
print "Image should be ", answer[0]
