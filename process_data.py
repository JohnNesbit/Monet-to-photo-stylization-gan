import numpy as np
import os
import PIL.Image as PIL
import pickle

#get data from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo
b_path = "E:/monet2photo/monet2photo/" # path to dir containing test and train folders


testA = np.array([])
testB = np.array([])
trainA = np.array([])
trainB = np.array([])

t_t = {
    0: "train",
    1: "test"
}

b_s = {
    0:"A",
    1:"B"

}


for m in range(2):
    path = b_path + t_t[m]
    for lol in range(2):
        path = path + b_s[lol]
        for i in os.listdir(path):
            rawi = PIL.open(path + "/" + i)
            bi, gi, ri = rawi.split()
            im = np.array(PIL.merge("RGB", (ri, gi, bi)).convert("RGB"))
            print(im.shape)
            exec(" = np.append(" + t_t[m] + b_s[lol] + ", im)")


print(testA.shape)
print(testB.shape)
print(trainA.shape)
print(trainB.shape)

pickle.dump(testA, open("b_train.pickle", "wb"))
pickle.dump(trainB, open("b_train.pickle", "wb"))
pickle.dump(testB, open("a_test.pickle", "wb"))
pickle.dump(trainA, open("b_test.pickle", "wb"))
