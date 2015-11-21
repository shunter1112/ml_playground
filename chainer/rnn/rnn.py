import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F
import six

model = FunctionSet(
    embed  = F.EmbedID(1000, 100),
    x_to_h = F.Linear(100,   50),
    h_to_h = F.Linear( 50,   50),
    h_to_y = F.Linear( 50, 1000),
)
optimizer = optimizers.SGD()
optimizer.setup(model)

def forward_one_step(h, cur_word, next_word, volatile=False):
    word = Variable(cur_word, volatile=volatile)
    t    = Variable(next_word, volatile=volatile)
    x    = F.tanh(model.embed(word))
    h    = F.tanh(model.x_to_h(x) + model.h_to_h(h))
    y    = model.h_to_y(h)
    loss = F.softmax_cross_entropy(y, t)
    return h, loss

def forward(x_list, volatile=False):
    h = Variable(np.zeros((1, 50), dtype=np.float32), volatile=volatile)
    loss = 0
    for cur_word, next_word in zip(x_list, x_list[1:]):
        h, new_loss = forward_one_step(h, cur_word, next_word, volatile=volatile)
        loss += new_loss
    return loss

