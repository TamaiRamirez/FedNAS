import numpy as np
import flwr_cifar10_enas.src.framework as fw

def lstm(x, prev_c, prev_h, w):
  """
  x: input data
  prev_c: previous cell state
  prev_h: previous hidden_state
  w: weights
  """
  
  """
  ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
  i, f, o, g = tf.split(ifog, 4, axis=1)
  i = tf.sigmoid(i)
  f = tf.sigmoid(f)
  o = tf.sigmoid(o)
  g = tf.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * tf.tanh(next_c)
  """
  i, f, o, g = fw.split(
    fw.matmul(
      fw.concat([x, prev_h], axis=1), 
      w),
    4,
    axis=1)
  """
  fw.concat(...): concatenate matrices, in this case, concatenate Input data and previous hidden state
  fw.matmul(...): Multiplies the matrices, in this case the concatenation matrix of x and prev_h with the weights
  fw.split(...): Splits a tensor value into a list of sub tensors.
  """
  input_gate = fw.sigmoid(i) * fw.tanh(g)
  forget_gate = fw.sigmoid(f)
  next_c = input_gate + forget_gate * prev_c #Cell state
  next_h = fw.sigmoid(o) * fw.tanh(next_c) #Output Gate
  return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
  """
  Generate a RNN by stacking various LSTM sequentially
  """
  next_c, next_h = [], []
  for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
    curr_c, curr_h = lstm(x if layer_id == 0 else next_h[-1], _c, _h, _w)
    next_c.append(curr_c)
    next_h.append(curr_h)
  return next_c, next_h
