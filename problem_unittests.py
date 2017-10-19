import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from attention_crop import *
import helper


k = 0.05
def _print_success_message():
    print('Tests Passed')


def test_get_corners(get_corners):
    with tf.Graph().as_default():
        params = np.array([[0.5, 0.5, 0.3],
                           [0.2, 0.2, 0.3],
                           [0.8, 0.8, 0.3],
                           [0.5, 0.2, 0.3],
                           [0.8, 0.2, 0.3]])
        params2 = np.array([0.5, 0.5, 0.3])
        boxes = get_corners(params)
        # boxes2 = get_corners(params2)
        assert isinstance(boxes, tf.Tensor), 'boxes is not a Tensor.'
        # print(boxes)
        assert boxes.get_shape().as_list() == [params.shape[0], 4], "boxes shape is not right."
        # assert boxes2.get_shape().as_list() == [params2.shape[0], 4]
        with tf.Session() as sess:
            correct_result = np.zeros((params.shape[0], 4), np.float32)
            correct_result[:, 0] = np.maximum(params[:, 0] - params[:, 2], 0)
            correct_result[:, 1] = np.maximum(params[:, 1] - params[:, 2], 0)
            correct_result[:, 2] = np.minimum(params[:, 0] + params[:, 2], 1)
            correct_result[:, 3] = np.minimum(params[:, 1] + params[:, 2], 1)
            assert np.allclose(boxes.eval(), correct_result)

    _print_success_message()


def test_attention_crop(attention_crop):
    with tf.Graph().as_default():
        image11 = np.zeros((20, 20))
        image12 = np.ones((20, 20))
        image21 = np.ones((20, 20))
        image22 = np.zeros((20, 20))
        image1 = np.concatenate((image11, image12), axis=1)
        image2 = np.concatenate((image21, image22), axis=1)
        image = np.concatenate((image1, image2))
        image = np.stack((image, image, image), axis=-1)
        image = np.expand_dims(image, axis=0)

        params = np.array([[0.3, 0.3, 0.3]], dtype=np.float32)
        output_size = 40
        image_out = attention_crop(image, params, output_size)
        # todo: a robust way to test the attention_crop correctness


def test_h(h):
    with tf.Graph().as_default():
        x = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        tfx = tf.constant(x)
        tfy = h(tfx)
        y = np.divide(1.0, np.add(1.0, np.exp(np.multiply(x, -k))))
        with tf.Session() as sess:
            assert np.allclose(y, tfy.eval())


def test_f(f):
    with tf.Graph().as_default():
        input1 = [5.0, 5, 3, 4, 2]
        tf_output1 = f(*input1)
        with tf.Session() as sess:
            print(tf_output1.eval())

def test_reduce_max():
    zeros = np.zeros((5, 5)) + 0.01
    ones = np.ones((5, 5))
    twos = np.ones((5, 5)) * 2

    values = np.stack([zeros, ones, twos], axis=2)
    values = np.expand_dims(values, axis=0)
    values2 = values * 2
    values = np.concatenate((values, values2), axis=0)
    channel_max = tf.reduce_max(values, axis=[1, 2])
    print(values)
    print("shape is {}".format(values.shape))
    with tf.Session() as sess:
        print("channel max is {}".format(channel_max.eval()))
        print("channel max shape is {}".format(channel_max.eval().shape))

def test_normalize_grad(normalize_grad):
    image_input = tf.Variable([[[2., 6, 15], [1, 2, 3]],
                               [[1, 2, 3], [1, 2, 3]],
                               ])
    init = tf.global_variables_initializer()
    normalized = normalize_grad(image_input)
    with tf.Session() as sess:
        sess.run(init)
        print(normalized.eval())


def test_variable():
    v = tf.Variable([1, 2])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(v.eval())

def test_get_derivatives(get_derivatives):
    with tf.Graph().as_default():
        param = [5, 8., 3]
        param = tf.convert_to_tensor(param)
        result = get_derivatives(param)
        with tf.Session() as sess:
            print(result.eval())
            print("shape is {}".format(result.eval().shape))


def test_multiply_broadcast():
    with tf.Graph().as_default():
        ones = tf.ones((2, 2))
        twos = ones * 2
        threes = ones * 3
        packed = tf.stack([ones, twos, threes], axis=2)
        twos_expanded = tf.expand_dims(twos, axis=-1)
        product = tf.multiply(packed, twos_expanded)
        with tf.Session() as sess:
            print(product.eval())


def test_attention_crop():
    image_input = tf.placeholder([None, 400, 400, 3], name="image_input")
    params = tf.Variable([0.5, 0.5, 0.3])
    output_size = 200
    def normalize(x):
        return x / 255 - 0.5

    def one_hot_encode(x):
        return tf.one_hot(x)

    image_output = attention_crop(image_input, params, output_size)
    fc = tf.contrib.layers.fully_connected(image_output)





def all_main():
    test_reduce_max()
    test_variable()
    test_get_corners(get_corners)
    test_attention_crop(attention_crop)
    test_h(h)
    test_f(f)
    test_normalize_grad(normalize_grad)

def main():
    test_get_derivatives(get_grad_f)
    get_grad_f2 = get_grad_f_helper(10.)
    test_get_derivatives(get_grad_f2)
    test_multiply_broadcast()

if __name__ == "__main__":
    main()





