import tensorflow as tf
import numpy as np
from tqdm import tqdm

import cifar10_input
from sap_model import SAPModel

num_iter = 100
num_img = 10000
batch_size = 10

cifar = cifar10_input.CIFAR10Data("../cifar10_data")

sess = tf.Session()
model = SAPModel("../models/standard/", tiny=False, mode='eval', sess=sess)

xs = tf.placeholder(tf.float32, (batch_size, 32, 32, 3))

avg = np.zeros((batch_size, 10))
correct = 0.0
m = 0

for img in tqdm(range(0, num_img, batch_size)):

    label = cifar.eval_data.ys[img : img + batch_size]
    image = cifar.eval_data.xs[img : img + batch_size]

    avg *= 0.0
    for i in range(num_iter):

        done = False
        while not done:
            try:
                model.init_mask(image, sess)
                logit = sess.run(model.pre_softmax, {model.x_input: image})
                done = True
            except:
                print('REPEAT\tImages:' + str(img) + '-' + str(img + batch_size) + '\tIter:' + str(i))
                done = False
        logit = logit.astype(np.float64)
        softmax = np.exp(logit)
        softmax /= np.sum(softmax, axis=1, keepdims=True)
        avg += softmax

    pred = np.argmax(avg, axis=1)
    for j in range(batch_size):
        if pred[j] == label[j]:
            correct += 1.0

    m += batch_size

    print('accuracy', correct / m)

