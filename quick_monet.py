from PIL import Image as PIL
import numpy as np
from cv2 import imwrite
from scipy.misc import imresize
from models import *
IMAGEPATH = input("path to image:")
epoch_num = input("epochs trained on trained model:")
BGR = input("type true if you got an unexpected error last time running this script:")
italy_array = PIL.open(IMAGEPATH)
if BGR:
    b, g, r = italy_array.split()
    im = np.array(PIL.merge("RGB", (r, g, b)).convert("RGB"))
else:
    im = np.array(italy_array.convert("RGB"))
italy_array = imresize(im, (256, 256, 3))


d_w1, d_b1 = ivar(shape=[256, 256, 3, 3], bshape=[1, 256, 256, 3],name="d_w1", bname="d_b1")
d_w2, d_b2 = ivar(shape=[128, 128, 3, 3], bshape=[1, 512, 512, 3],name="d_w2", bname="d_b2")
d_w3, d_b3 = ivar(shape=[256, 256, 3, 3], bshape=[1, 256, 256, 3],name="d_w3", bname="d_b3")
d_w4, d_b4 = ivar(shape=[256, 256, 3, 3], bshape=[1, 256, 256, 3],name="d_w4", bname="d_b4")
d_w5, d_b5 = ivar(shape=[64, 64, 3, 3], bshape=[1, 128, 128, 3],name="d_w5", bname="d_b5")
d_w6, d_b6 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="d_w6", bname="d_b6")
d_w7, d_b7 = ivar(shape=[64, 64, 3, 3], bshape=[1, 128, 128, 3],name="d_w7", bname="d_b7")
d_w8, d_b8 = ivar(shape=[64, 64, 3, 3], bshape=[1, 128, 128, 3],name="d_w8", bname="d_b8")
d_w9, d_b9 = ivar(shape=[128, 128, 3, 3], bshape=[1, 64, 64, 3],name="d_w9", bname="d_b9")
d_w10, d_b10 = ivar(shape=[64, 64, 3, 3], bshape=[1, 128, 128, 3],name="d_w10", bname="d_b10")
d_w11, d_b11 = ivar(shape=[64, 64, 3, 3], bshape=[1, 64, 64, 3],name="d_w11", bname="d_b11")
d_w12, d_b12 = ivar(shape=[64, 64, 3, 3], bshape=[1, 32, 32, 3],name="d_w12", bname="d_b12")
d_w13, d_b13 = ivar(shape=[64, 64, 3, 3], bshape=[1, 16, 16, 3],name="d_w13", bname="d_b13")
d_w14, d_b14 = ivar(shape=[64, 64, 3, 3], bshape=[1, 16, 16, 3],name="d_w14", bname="d_b14")
d_w15, d_b15 = ivar(shape=[64, 64, 3, 3], bshape=[1, 8, 8, 3],name="d_w15", bname="d_b15")
d_w16, d_b16 = ivar(shape=[32, 32, 3, 3], bshape=[1, 4, 4, 3],name="d_w16", bname="d_b16")
d_w17, d_b17 = ivar(shape=[16, 16, 3, 3], bshape=[1, 2, 2, 3],name="d_w17", bname="d_b17")
d_w18, d_b18 = ivar(shape=[12, 1], bshape=[1],name="d_w18", bname="d_b18")

id_w1, id_b1 = ivar(shape=[256, 256, 3, 3], bshape=[1, 256, 256, 3],name="id_w1", bname="id_b1")
id_w2, id_b2 = ivar(shape=[128, 128, 3, 3], bshape=[1, 512, 512, 3],name="id_w2", bname="id_b2")
id_w3, id_b3 = ivar(shape=[256, 256, 3, 3], bshape=[1, 256, 256, 3],name="id_w3", bname="id_b3")
id_w4, id_b4 = ivar(shape=[256, 256, 3, 3], bshape=[1, 256, 256, 3],name="id_w4", bname="id_b4")
id_w5, id_b5 = ivar(shape=[64, 64, 3, 3], bshape=[1, 128, 128, 3],name="id_w5", bname="id_b5")
id_w6, id_b6 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="id_w6", bname="id_b6")
id_w7, id_b7 = ivar(shape=[64, 64, 3, 3], bshape=[1, 128, 128, 3],name="id_w7", bname="id_b7")
id_w8, id_b8 = ivar(shape=[64, 64, 3, 3], bshape=[1, 128, 128, 3],name="id_w8", bname="id_b8")
id_w9, id_b9 = ivar(shape=[128, 128, 3, 3], bshape=[1, 64, 64, 3],name="id_w9", bname="id_b9")
id_w10, id_b10 = ivar(shape=[64, 64, 3, 3], bshape=[1, 128, 128, 3],name="id_w10", bname="id_b10")
id_w11, id_b11 = ivar(shape=[64, 64, 3, 3], bshape=[1, 64, 64, 3],name="id_w11", bname="id_b11")
id_w12, id_b12 = ivar(shape=[64, 64, 3, 3], bshape=[1, 32, 32, 3],name="id_w12", bname="id_b12")
id_w13, id_b13 = ivar(shape=[64, 64, 3, 3], bshape=[1, 16, 16, 3],name="id_w13", bname="id_b13")
id_w14, id_b14 = ivar(shape=[64, 64, 3, 3], bshape=[1, 16, 16, 3],name="id_w14", bname="id_b14")
id_w15, id_b15 = ivar(shape=[64, 64, 3, 3], bshape=[1, 8, 8, 3],name="id_w15", bname="id_b15")
id_w16, id_b16 = ivar(shape=[32, 32, 3, 3], bshape=[1, 4, 4, 3],name="id_w16", bname="id_b16")
id_w17, id_b17 = ivar(shape=[16, 16, 3, 3], bshape=[1, 2, 2, 3],name="id_w17", bname="id_b17")
id_w18, id_b18 = ivar(shape=[12, 1], bshape=[1],name="id_w18", bname="id_b18")


g_w1, g_b1 = ivar(shape=[16, 16, 3, 3], bshape=[1, 256, 256, 3],name="g_w1", bname="g_b1")
g_w2, g_b2 = ivar(shape=[32, 32, 3, 3], bshape=[1, 512, 512, 3],name="g_w2", bname="g_b2")
g_w3, g_b3 = ivar(shape=[32, 32, 3, 3], bshape=[1, 256, 256, 3],name="g_w3", bname="g_b3")
g_w4, g_b4 = ivar(shape=[64, 64, 3, 3], bshape=[1, 512, 512, 3],name="g_w4", bname="g_b4")
g_w5, g_b5 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="g_w5", bname="g_b5")
g_w6, g_b6 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="g_w6", bname="g_b6")
g_w7, g_b7 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="g_w7", bname="g_b7")
g_w8, g_b8 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="g_w8", bname="g_b8")
g_w9, g_b9 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="g_w9", bname="g_b9")
g_w10, g_b10 = ivar(shape=[256, 256, 3, 3], bshape=[1, 256, 256, 3],name="g_w10", bname="g_b10")
g_w11, g_b11 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="g_w11", bname="g_b11")
g_w12, g_b12 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="g_w12", bname="g_b12")
g_w13, g_b13 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="g_w13", bname="g_b13")
g_w14, g_b14 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="g_w14", bname="g_b14")
g_w15, g_b15 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="g_w15", bname="g_b15")
g_w16, g_b16 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="g_w16", bname="g_b16")
g_w17, g_b17 = ivar(shape=[32, 32, 3, 3], bshape=[1, 256, 256, 3],name="g_w17", bname="g_b17")
g_w18, g_b18 = ivar(shape=[32, 32, 3, 3], bshape=[1, 256, 256, 3],name="g_w18", bname="g_b18")

ig_w1, ig_b1 = ivar(shape=[16, 16, 3, 3], bshape=[1, 256, 256, 3],name="ig_w1", bname="ig_b1")
ig_w2, ig_b2 = ivar(shape=[32, 32, 3, 3], bshape=[1, 512, 512, 3],name="ig_w2", bname="ig_b2")
ig_w3, ig_b3 = ivar(shape=[32, 32, 3, 3], bshape=[1, 256, 256, 3],name="ig_w3", bname="ig_b3")
ig_w4, ig_b4 = ivar(shape=[64, 64, 3, 3], bshape=[1, 512, 512, 3],name="ig_w4", bname="ig_b4")
ig_w5, ig_b5 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="ig_w5", bname="ig_b5")
ig_w6, ig_b6 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="ig_w6", bname="ig_b6")
ig_w7, ig_b7 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="ig_w7", bname="ig_b7")
ig_w8, ig_b8 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="ig_w8", bname="ig_b8")
ig_w9, ig_b9 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="ig_w9", bname="ig_b9")
ig_w10, ig_b10 = ivar(shape=[256, 256, 3, 3], bshape=[1, 256, 256, 3],name="ig_w10", bname="ig_b10")
ig_w11, ig_b11 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="ig_w11", bname="ig_b11")
ig_w12, ig_b12 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="ig_w12", bname="ig_b12")
ig_w13, ig_b13 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="ig_w13", bname="ig_b13")
ig_w14, ig_b14 = ivar(shape=[128, 128, 3, 3], bshape=[1, 256, 256, 3],name="ig_w14", bname="ig_b14")
ig_w15, ig_b15 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="ig_w15", bname="ig_b15")
ig_w16, ig_b16 = ivar(shape=[64, 64, 3, 3], bshape=[1, 256, 256, 3],name="ig_w16", bname="ig_b16")
ig_w17, ig_b17 = ivar(shape=[32, 32, 3, 3], bshape=[1, 256, 256, 3],name="ig_w17", bname="ig_b17")
ig_w18, ig_b18 = ivar(shape=[32, 32, 3, 3], bshape=[1, 256, 256, 3],name="ig_w18", bname="ig_b18")


def ab_gen(x):
    x = tf.reshape(x, [1, 256, 256, 3])
    o1 = leaky_relu(tf.nn.conv2d(x, g_w1, strides=[1, 1, 1, 1], padding="SAME") + g_b1)

    o2 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o1, g_w2, output_shape=[1, 512, 512, 3], strides=[1, 2, 2, 1]) + g_b2))

    o3 = tf.nn.conv2d(o2, g_w3, strides=[1, 2, 2, 1], padding="SAME") + g_b3
    o3 = leaky_relu(instance_norm(o3) + o1)

    o4 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o3, g_w4, output_shape=[1, 512, 512, 3], strides=[1, 2, 2, 1]) + g_b4))

    o5 = tf.nn.conv2d(o4, g_w5, strides=[1, 2, 2, 1], padding="SAME") + g_b5
    o5 = leaky_relu(instance_norm(o5) + o3)

    o6 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o5, g_w6, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b6))

    o7 = tf.nn.conv2d(o6, g_w7, strides=[1, 1, 1, 1], padding="SAME") + g_b7
    o7 = leaky_relu(instance_norm(o7) + o5)

    o8 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o7, g_w8, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b8))

    o9 = tf.nn.conv2d(o8, g_w9, strides=[1, 1, 1, 1], padding="SAME") + g_b9
    o9 = leaky_relu(instance_norm(o9) + o7)

    o10 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o9, g_w10, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b10))

    o11 = tf.nn.conv2d(o10, g_w11, strides=[1, 1, 1, 1], padding="SAME")+ g_b11
    o11 = leaky_relu(instance_norm(o11) + o9)

    o12 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o11, g_w12, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b12))

    o13 = tf.nn.conv2d(o12, g_w13, strides=[1, 1, 1, 1], padding="SAME") + g_b13
    o13 = leaky_relu(instance_norm(o13) + o11)

    o14 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o13, g_w14, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b14))

    o15 = tf.nn.conv2d(o14, g_w15, strides=[1, 1, 1, 1], padding="SAME") + g_b15
    o15 = leaky_relu(instance_norm(o15) + o13)

    o16 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o15, g_w16, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b16))

    o17 = tf.nn.conv2d(o16, g_w17, strides=[1, 1, 1, 1], padding="SAME") + g_b17
    o17 = leaky_relu(instance_norm(o17))

    o18 = leaky_relu(instance_norm(tf.nn.conv2d(o17, g_w18, strides=[1, 1, 1, 1], padding="SAME") + g_b18))

    return o18


def ba_gen(x):
    x = tf.reshape(x, [1, 256, 256, 3])
    o1 = leaky_relu(tf.nn.conv2d(x, ig_w1, strides=[1, 1, 1, 1], padding="SAME") + ig_b1)

    o2 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o1, ig_w2, output_shape=[1, 512, 512, 3], strides=[1, 2, 2, 1]) + ig_b2))

    o3 = tf.nn.conv2d(o2, ig_w3, strides=[1, 2, 2, 1], padding="SAME") + ig_b3
    o3 = leaky_relu(instance_norm(o3) + o1)

    o4 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o3, ig_w4, output_shape=[1, 512, 512, 3], strides=[1, 2, 2, 1]) + ig_b4))

    o5 = tf.nn.conv2d(o4, ig_w5, strides=[1, 2, 2, 1], padding="SAME") + ig_b5
    o5 = leaky_relu(instance_norm(o5) + o3)

    o6 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o5, ig_w6, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + ig_b6))

    o7 = tf.nn.conv2d(o6, ig_w7, strides=[1, 1, 1, 1], padding="SAME") + ig_b7
    o7 = leaky_relu(instance_norm(o7) + o5)

    o8 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o7, ig_w8, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + ig_b8))

    o9 = tf.nn.conv2d(o8, ig_w9, strides=[1, 1, 1, 1], padding="SAME") + ig_b9
    o9 = leaky_relu(instance_norm(o9) + o7)

    o10 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o9, ig_w10, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b10))

    o11 = tf.nn.conv2d(o10, ig_w11, strides=[1, 1, 1, 1], padding="SAME")+ ig_b11
    o11 = leaky_relu(instance_norm(o11) + o9)

    o12 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o11, ig_w12, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b12))

    o13 = tf.nn.conv2d(o12, ig_w13, strides=[1, 1, 1, 1], padding="SAME") + ig_b13
    o13 = leaky_relu(instance_norm(o13) + o11)

    o14 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o13, ig_w14, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + ig_b14))

    o15 = tf.nn.conv2d(o14, ig_w15, strides=[1, 1, 1, 1], padding="SAME") + ig_b15
    o15 = leaky_relu(instance_norm(o15) + o13)

    o16 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o15, ig_w16, output_shape=[1, 256, 256, 3], strides=[1, 1, 1, 1]) + g_b16))

    o17 = tf.nn.conv2d(o16, ig_w17, strides=[1, 1, 1, 1], padding="SAME") + ig_b17
    o17 = leaky_relu(instance_norm(o17))

    o18 = leaky_relu(instance_norm(tf.nn.conv2d(o17, ig_w18, strides=[1, 1, 1, 1], padding="SAME") + ig_b18))

    return o18


def a_descrim(x):
    x = tf.reshape(x, [1, 256, 256, 3])
    o1 = leaky_relu(tf.nn.conv2d(x, id_w1, strides=[1, 1, 1, 1], padding="SAME") + id_b1)

    o2 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o1, id_w2, output_shape=[1, 512, 512, 3], strides=[1, 2, 2, 1]) + id_b2))

    o3 = tf.nn.conv2d(o2, id_w3, strides=[1, 2, 2, 1], padding="SAME") + id_b3
    o3 = leaky_relu(instance_norm(o3) + o1)

    o4 = leaky_relu(instance_norm(tf.nn.conv2d(o3, id_w4, padding="SAME", strides=[1, 1, 1, 1]) + id_b4))

    o5 = tf.nn.conv2d(o4, id_w5, strides=[1, 2, 2, 1], padding="SAME") + id_b5
    o5 = leaky_relu(instance_norm(o5))

    o6 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o5, id_w6, output_shape=[1, 256, 256, 3], strides=[1, 2, 2, 1]) + id_b6))

    o7 = tf.nn.conv2d(o6, id_w7, strides=[1, 2, 2, 1], padding="SAME") + id_b7
    o7 = leaky_relu(instance_norm(o7) + o5)

    o8 = leaky_relu(instance_norm(tf.nn.conv2d(o7, id_w8, padding="SAME", strides=[1, 1, 1, 1]) + id_b8))

    o9 = tf.nn.conv2d(o8, id_w9, strides=[1, 2, 2, 1], padding="SAME") + id_b9
    o9 = leaky_relu(instance_norm(o9))

    o10 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o9, id_w10, output_shape=[1, 128, 128, 3], strides=[1, 2, 2, 1]) + id_b10))

    o11 = tf.nn.conv2d(o10, id_w11, strides=[1, 2, 2, 1], padding="SAME")+ id_b11
    o11 = leaky_relu(instance_norm(o11) + o9)

    o12 = tf.nn.conv2d(o11, id_w12, strides=[1, 2, 2, 1], padding="SAME") + id_b12
    o12 = leaky_relu(instance_norm(o12))

    o13 = tf.nn.conv2d(o12, id_w13, strides=[1, 2, 2, 1], padding="SAME") + id_b13
    o13 = leaky_relu(instance_norm(o13))

    o14 = leaky_relu(instance_norm(tf.nn.conv2d(o13, id_w14, padding="SAME", strides=[1, 1, 1, 1]) + id_b14))

    o15 = tf.nn.conv2d(o14, id_w15, strides=[1, 2, 2, 1], padding="SAME") + id_b15
    o15 = leaky_relu(instance_norm(o15))

    o16 = tf.nn.conv2d(o15, id_w16, strides=[1, 2, 2, 1], padding="SAME") + id_b16
    o16 = leaky_relu(instance_norm(o16))

    o17 = tf.nn.conv2d(o16, id_w17, strides=[1, 2, 2, 1], padding="SAME") + id_b17
    o17 = leaky_relu(instance_norm(o17))
    o17 = tf.reshape(o17, [1, 12])
    o18 = leaky_relu(tf.matmul(o17, id_w18) + id_b18)

    return tf.reshape(o18, shape=[1])


def b_descrim(x):
    x = tf.reshape(x, [1, 256, 256, 3])
    o1 = leaky_relu(tf.nn.conv2d(x, d_w1, strides=[1, 1, 1, 1], padding="SAME") + d_b1)

    o2 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o1, d_w2, output_shape=[1, 512, 512, 3], strides=[1, 2, 2, 1]) + d_b2))

    o3 = tf.nn.conv2d(o2, d_w3, strides=[1, 2, 2, 1], padding="SAME") + d_b3
    o3 = leaky_relu(instance_norm(o3) + o1)

    o4 = leaky_relu(instance_norm(tf.nn.conv2d(o3, id_w4, padding="SAME", strides=[1, 1, 1, 1]) + id_b4))

    o5 = tf.nn.conv2d(o4, d_w5, strides=[1, 2, 2, 1], padding="SAME") + d_b5
    o5 = leaky_relu(instance_norm(o5))

    o6 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o5, d_w6, output_shape=[1, 256, 256, 3], strides=[1, 2, 2, 1]) + d_b6))

    o7 = tf.nn.conv2d(o6, d_w7, strides=[1, 2, 2, 1], padding="SAME") + d_b7
    o7 = leaky_relu(instance_norm(o7) + o5)

    o8 = leaky_relu(instance_norm(tf.nn.conv2d(o7, id_w8, padding="SAME", strides=[1, 1, 1, 1]) + id_b8))

    o9 = tf.nn.conv2d(o8, d_w9, strides=[1, 2, 2, 1], padding="SAME") + d_b9
    o9 = leaky_relu(instance_norm(o9))

    o10 = leaky_relu(instance_norm(tf.nn.conv2d_transpose(o9, d_w10, output_shape=[1, 128, 128, 3], strides=[1, 2, 2, 1]) + d_b10))

    o11 = tf.nn.conv2d(o10, d_w11, strides=[1, 2, 2, 1], padding="SAME")+ d_b11
    o11 = leaky_relu(instance_norm(o11) + o9)

    o12 = tf.nn.conv2d(o11, d_w12, strides=[1, 2, 2, 1], padding="SAME") + d_b12
    o12 = leaky_relu(instance_norm(o12))

    o13 = tf.nn.conv2d(o12, d_w13, strides=[1, 2, 2, 1], padding="SAME") + d_b13
    o13 = leaky_relu(instance_norm(o13))

    o14 = leaky_relu(instance_norm(tf.nn.conv2d(o13, id_w14, padding="SAME", strides=[1, 1, 1, 1]) + id_b14))

    o15 = tf.nn.conv2d(o14, d_w15, strides=[1, 2, 2, 1], padding="SAME") + d_b15
    o15 = leaky_relu(instance_norm(o15))

    o16 = tf.nn.conv2d(o15, d_w16, strides=[1, 2, 2, 1], padding="SAME") + d_b16
    o16 = leaky_relu(instance_norm(o16))

    o17 = tf.nn.conv2d(o16, d_w17, strides=[1, 2, 2, 1], padding="SAME") + d_b17
    o17 = leaky_relu(instance_norm(o17))
    o17 = tf.reshape(o17, [1, 12])
    o18 = leaky_relu(tf.matmul(o17, d_w18) + d_b18)

    return tf.reshape(o18, shape=[1])


sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "models/pretrained_mon" + epoch_num + ".ckpt")
italy_array = ba_gen(italy_array).eval()
imwrite("resizednew.jpg", italy_array)