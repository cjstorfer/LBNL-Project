# pylint: disable=C,R,no-member
import tensorflow as tf
import numpy as np
import layers_normal as nn


def dihedral(x, i):
    if len(x.shape) == 3:
        if i & 4:
            y = np.transpose(x, (1, 0, 2))
        else:
            y = x.copy()

        if i&3 == 0:
            return y
        if i&3 == 1:
            return y[:, ::-1]
        if i&3 == 2:
            return y[::-1, :]
        if i&3 == 3:
            return y[::-1, ::-1]

    if len(x.shape) == 4:
        if i & 4:
            y = np.transpose(x, (0, 2, 1, 3))
        else:
            y = x.copy()

        if i&3 == 0:
            return y
        if i&3 == 1:
            return y[:, :, ::-1]
        if i&3 == 2:
            return y[:, ::-1, :]
        if i&3 == 3:
            return y[:, ::-1, ::-1]

def res_layer(x, f_out=None, w=3, n=1):
    f_in = x.get_shape().as_list()[3]
    assert w % 2 == 1
    if f_out is None:
        f_out = f_in

    with tf.name_scope("res_layer"):
        b = (w // 2) * n
        s = x[:, b:-b, b:-b, :] # VALID padding
        if f_in != f_out:
            s = nn.convolution(s, f_out, w=1, name="shortcut", activation=None)

        if n == 1:
            x = nn.convolution(x, f_out, w=w, activation=None)
        else:
            x = nn.convolution(x, f_out, w=w)
            for _ in range(n - 2):
                x = nn.convolution(x, f_out, w=w)
            x = nn.convolution(x, f_out, w=w, activation=None)

        x = nn.relu(0.7071067811865475 * (s + x))
        return x

def summary_images(x, name):
    for i in range(min(4, x.get_shape().as_list()[3])):
        tf.summary.image("{}-{}".format(name, i), x[:, :, :, i:i+1])

class CNN:
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.tfx = None
        self.tfy = None
        self.tfp = None
        self.xent = None
        self.tftrain_step = None
        self.tfkp = None
        self.tfacc = None
        self.train_counter = 0
        self.test = None
        self.embedding_input = None


    def NN(self, x):
        assert x.get_shape().as_list()[:3] == [None, 101, 101]
        summary_images(x, "layer0")
        x = nn.convolution(x, 16, w=2) # 100
        summary_images(x, "layer1")

        ########################################################################
        assert x.get_shape().as_list() == [None, 100, 100, 16]
        x = nn.batch_normalization(x, self.tfacc)
        x = res_layer(x, n=2) # 96
        x = nn.batch_normalization(x, self.tfacc)
        x = res_layer(x, n=2) # 92
        summary_images(x, "layer5")
        x = nn.max_pool(x)

        ########################################################################
        assert x.get_shape().as_list() == [None, 46, 46, 16]
        x = nn.batch_normalization(x, self.tfacc)
        x = res_layer(x, 32, n=3) # 40
        summary_images(x, "layer8")
        x = nn.batch_normalization(x, self.tfacc)
        x = res_layer(x, n=2) # 36
        summary_images(x, "layer10")
        x = nn.max_pool(x)
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        ########################################################################
        assert x.get_shape().as_list() == [None, 18, 18, 32]
        x = res_layer(x, 64, n=2) # 14
        summary_images(x, "layer12")
        x = nn.batch_normalization(x, self.tfacc)
        x = res_layer(x, 92, n=2) # 10
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)
        x = res_layer(x, 128, n=3) # 4
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        ########################################################################
        assert x.get_shape().as_list() == [None, 4, 4, 128]
        x = nn.convolution(x, 1024, w=4)

        ########################################################################
        assert x.get_shape().as_list() == [None, 1, 1, 1024]
        x = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])
        self.embedding_input = x

        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        x = nn.fullyconnected(x, 1024)
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        x = nn.fullyconnected(x, 1024)
        x = nn.batch_normalization(x, self.tfacc)
        self.test = x

        x = nn.fullyconnected(x, 1, activation=None)
        return x

    ########################################################################
    def create_architecture(self, bands):
        self.tfkp = tf.placeholder_with_default(tf.constant(1.0, tf.float32), [], name="kp")
        self.tfacc = tf.placeholder_with_default(tf.constant(0.0, tf.float32), [], name="acc")
        x = self.tfx = tf.placeholder(tf.float32, [None, 101, 101, bands], name="input")
        # mean = 0 and std = 1

        with tf.name_scope("nn"):
            x = self.NN(x)

        ########################################################################
        assert x.get_shape().as_list() == [None, 1]
        self.tfp = tf.nn.sigmoid(tf.reshape(x, [-1]))

        with tf.name_scope("xent"):
            self.tfy = tf.placeholder(tf.float32, [None])
            xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.tfy, [-1, 1]))
            # [None, 1]
            self.xent = tf.reduce_mean(xent)

        with tf.name_scope("train"):
            self.tftrain_step = tf.train.AdamOptimizer(1e-4).minimize(self.xent)

    @staticmethod
    def split_test_train(path):
        # extract the files names and split them into test and train set
        import os
        files = ['{}/{}'.format(path, f) for f in sorted(os.listdir(path))]
        return files[:3000], files[3000:]

    @staticmethod
    def load(files):
        # load some files into numpy array ready to be eaten by tensorflow
        xs = np.stack([np.load(f)['image'] for f in files])
        return CNN.prepare(xs)

    @staticmethod
    def prepare(images):
        images[images == 100] = 0.0
        if images.shape[-1] == 1:
            images = (images - 4.337e-13) / 5.504e-12
        elif images.shape[-1] == 4:
            images = (images - 1.685e-12) / 5.122e-11
        else:
            print("No statistics to prepare this kind of data")
        return images

    @staticmethod
    def batch(files, labels):
        # pick randomly some files, load them and make augmentation
        id0 = np.where(labels == 0)[0]
        id1 = np.where(labels == 1)[0]

        k = 15
        idn = np.random.choice(id0, k, replace=False)
        idp = np.random.choice(id1, k, replace=False)

        xs = CNN.load([files[i] for i in idp] + [files[i] for i in idn])
        ys = np.concatenate((labels[idp], labels[idn]))

        for i in range(len(xs)):
            s = np.random.uniform(0.8, 1.2)
            u = np.random.uniform(-0.1, 0.1)
            xs[i] = dihedral(xs[i], np.random.randint(8)) * s + u

        return xs, ys

    def train(self, session, xs, ys, options=None, run_metadata=None, tensors=None):
        if tensors is None:
            tensors = []

        acc = 0.6 ** (self.train_counter / 1000.0)
        kp = 0.5 + 0.5 * 0.5 ** (self.train_counter / 2000.0)

        output = session.run([self.tftrain_step, self.xent] + tensors,
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: kp, self.tfacc: acc},
            options=options, run_metadata=run_metadata)

        self.train_counter += 1
        return output[1], output[2:]

    def predict_naive(self, session, images):
        return session.run(self.tfp, feed_dict={self.tfx: images})

    def predict_naive_xentropy(self, session, images, labels):
        return session.run([self.tfp, self.xent], feed_dict={self.tfx: images, self.tfy: labels})

    def predict(self, session, images):
        # exploit symmetries to make better predictions
        ps = self.predict_naive(session, images)

        for i in range(1, 8):
            ps *= self.predict_naive(session, dihedral(images, i))

        return ps

    def predict_xentropy(self, session, images, labels):
        # exploit symmetries to make better predictions
        ps, xent = self.predict_naive_xentropy(session, images, labels)

        for i in range(1, 8):
            ps *= self.predict_naive(session, dihedral(images, i))

        return ps, xent
