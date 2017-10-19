import tensorflow as tf


def get_corners(param):
    param = tf.convert_to_tensor(param, dtype=tf.float32)
    tx, ty, tl = param[:, 0], param[:, 1], param[:, 2]
    tx_tl = tf.maximum(tx - tl, 0.)
    ty_tl = tf.maximum(ty - tl, 0.)
    tx_br = tf.minimum(tx + tl, 1.)
    ty_br = tf.minimum(ty + tl, 1.)
    return tf.transpose(tf.convert_to_tensor([tx_tl, ty_tl, tx_br, ty_br]))


def attention_crop(images, params, out_image_size, name=None):
    with tf.name_scope(name, "AttentionCrop", [images, params, out_image_size]) as scope:
        images = tf.convert_to_tensor(images, name="images")
        params = tf.convert_to_tensor(params, name="params")
        params = params / images.get_shape().as_list()[1]
        crop_size = tf.convert_to_tensor([out_image_size, out_image_size], name="crop_size")
        boxes = get_corners(params)
        box_ind = tf.scan(lambda a, _: a + 1, params, -1)
        out_images = tf.image.crop_and_resize(images, boxes, box_ind, crop_size)
        return out_images

k = 0.05
h = lambda x: tf.divide(1.0, tf.add(1.0, tf.exp(tf.multiply(x, -k))))
def diff_h(x):
    return k * tf.exp(tf.multiply(-k, x)) / ((1 + tf.exp(-k * x)) *
                                             (1 + tf.exp(-k * x)))


def f(tx, ty, tl, x, y):
    tx = tf.convert_to_tensor(tx, dtype=tf.float32)
    ty = tf.convert_to_tensor(ty, dtype=tf.float32)
    tl = tf.convert_to_tensor(tl, dtype=tf.float32)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return (h(x - (tx - tl)) -
            h(x - (tx + tl))
            ) * (
        h(y - (ty - tl)) -
        h(y - (ty + tl))
    )


def diff_f_tx(tx, ty, tl, x, y):
    return (diff_h(x - (tx - tl)) -
            diff_h(x - (tx + tl))
            ) * (
        h(y - (ty - tl)) -
        h(y - (ty + tl))
    )


def diff_f_ty(tx, ty, tl, x, y):
    return (diff_h(y - (ty - tl)) -
            diff_h(y - (ty + tl))
            ) * (
        h(x - (tx - tl)) -
        h(x - (tx + tl))
    )

def diff_f_tl(tx, ty, tl, x, y):
    return -((diff_h(y - (ty - tl)) + diff_h(y - (ty + tl))) * (h(x - (tx - tl)) - h(x - (tx + tl))) + (
    diff_h(x - (tx - tl)) + diff_h(x - (tx + tl))) * (h(y - (ty - tl)) - h(y - (ty + tl)))) + 0.005


def normalize_grad(grad):
    height, width, channel = grad.get_shape().as_list()
    channel_max = tf.reduce_max(grad, axis=[0, 1])
    c1 = tf.cond(channel_max[0] > 0., lambda: tf.multiply(tf.ones([height, width]), channel_max[0]),
                 lambda: tf.ones([height, width]))
    c2 = tf.cond(channel_max[1] > 0., lambda: tf.multiply(tf.ones([height, width]), channel_max[1]),
                 lambda: tf.ones([height, width]))
    c3 = tf.cond(channel_max[2] > 0., lambda: tf.multiply(tf.ones([height, width]), channel_max[2]),
                 lambda: tf.ones([height, width]))
    c_max = tf.stack([c1, c2, c3], axis=2)
    normalized = grad / c_max
    sumed = tf.reduce_sum(normalized, axis=2, keep_dims=True)
    return sumed


def get_grad_f(param): ## compute on the region
    tx, ty, tl = param[0], param[1], param[2]
    # create a tensor first
    indices_x = tf.range(tx - tl, tx + tl + 1)
    indices_y = tf.range(ty - tl, ty + tl + 1)

    bottom2top_x = tf.map_fn(lambda i: i, indices_x)
    layer_x = tf.map_fn(lambda i: bottom2top_x, indices_x)
    bottom2top_y = tf.map_fn(lambda i: i, indices_y)
    layer_y = tf.map_fn(lambda i: bottom2top_y, indices_y)
    layer_y = tf.transpose(layer_y)
    indices = tf.stack([layer_x, layer_y], axis=2)
    gradients_f_tx = tf.map_fn(lambda row:
                               tf.map_fn(lambda col: diff_f_tx(tx, ty, tl, col[0], col[1]), row),
                               indices)
    gradients_f_ty = tf.map_fn(lambda row:
                               tf.map_fn(lambda col: diff_f_ty(tx, ty, tl, col[0], col[1]), row),
                               indices)
    gradients_f_tl = tf.map_fn(lambda row:
                               tf.map_fn(lambda col: diff_f_tl(tx, ty, tl, col[0], col[1]), row),
                               indices)

    return tf.stack([gradients_f_tx, gradients_f_ty, gradients_f_tl], axis=2)


#    for x in tf.range(tx - tl, tx + tl + 1):
#        for y in tf.range(ty - tl, ty + tl + 1):
#            diff_f_tx(tx, ty, tl, x, y)
#            diff_f_ty(tx, ty, tl, x, y)
#            diff_f_tl(tx, ty, tl, x, y)
#
def get_grad_f_helper(out_size):
    def get_grad_f2(param):
        tx, ty, tl = param[0], param[1], param[2]
        ranges = tf.range(out_size)
        bottom2top = tf.map_fn(lambda i: i, ranges)
        layer_x = tf.map_fn(lambda i: bottom2top, ranges)
        layer_y = tf.transpose(layer_x)
        layer_x = tx - tl + 2 * tl / out_size * layer_x
        layer_y = ty - tl + 2 * tl / out_size * layer_y
        indices = tf.stack([layer_x, layer_y], axis=2)
        gradients_f_tx = tf.map_fn(lambda row:
                                   tf.map_fn(lambda col: diff_f_tx(tx, ty, tl, col[0], col[1]), row),
                                   indices)
        gradients_f_ty = tf.map_fn(lambda row:
                                   tf.map_fn(lambda col: diff_f_ty(tx, ty, tl, col[0], col[1]), row),
                                   indices)
        gradients_f_tl = tf.map_fn(lambda row:
                                   tf.map_fn(lambda col: diff_f_tl(tx, ty, tl, col[0], col[1]), row),
                                   indices)

        return tf.stack([gradients_f_tx, gradients_f_ty, gradients_f_tl], axis=2)
    return get_grad_f2


@tf.RegisterGradient("AttentionCrop")
def _attention_crop_grad(op, grad):
    images = op.inputs[0]
    params = op.inputs[1]
    out_image_size = op.inputs[2]

    top_diff = grad
    top_data = op.outputs[0]
    grad_to_params = tf.zeros_like(params)
    input_image_size = images.get_shape().as_list[2]
    # count1, count2, count3 ## what are these guys
    # top is exactly an image of three channels
    grad2 = tf.abs(grad)
    grad_normalized = tf.map_fn(normalize_grad, grad2)
    grad_normalized = grad_normalized * 0.0000001
    out_size = grad.get_shape().as_list[2]
    assert out_image_size == out_size
    get_grad_f2 = get_grad_f_helper(out_size)
    derivatives_f = tf.map_fn(get_grad_f2, params)
    grad_product = derivatives_f * grad_normalized
    grad_to_params = tf.reduce_sum(grad_product, axis=[0, 1])

    return tf.zeros_like(images), grad_to_params, tf.zeros_like(out_image_size)


if __name__ == "__main__":
    pass