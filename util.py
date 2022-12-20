# Imports
import numpy as np
import tensorflow as tf
import cv2

# Transforms rgb input images to YIQ color space then extracts luminance channel (Y) from each image.
# Optionally adjusts histogram of style luminance to match that of the content before returning both luminance channels.
def extract_luminances(rgb_content, rgb_style, match=True):
    # Translating all rgb images to yiq color space
    yiq_content = tf.image.rgb_to_yiq(rgb_content)
    y_content = tf.split(yiq_content, yiq_content.shape[2], 2)[0]
    yiq_style = tf.image.rgb_to_yiq(rgb_style)
    y_style = tf.split(yiq_style, yiq_style.shape[2], 2)[0]

    if match:
        # Update style image's Y (luminance) values
        # Formula from https://arxiv.org/pdf/1606.05897.pdf (luminance-only transfer section)
        std_ratio = np.std(y_content) / np.std(y_style)
        style_mean = np.mean(y_style)
        content_mean = np.mean(y_content)
        y_style =  tf.map_fn(lambda pix: (std_ratio*(pix-style_mean)+content_mean), y_style)

    return y_content, y_style

# Recolor generated image
def remerge(y_gen, rgb_content):
    yiq_content = tf.image.rgb_to_yiq(rgb_content)
    iq_content = tf.split(yiq_content, yiq_content.shape[2], 2)[1:]
    yiq_gen = tf.concat([y_gen, iq_content[0], iq_content[1]], axis=2)
    rgb_gen = tf.image.yiq_to_rgb(yiq_gen)
    return rgb_gen

def gram_matrix(input, shift=True):
    if len(input.shape) == 4:
        input = input[0]
    if shift:
        input = input-1
    result = tf.matmul(input, input, transpose_a=True)
    return result #/ input.shape[0]

# Takes a color image, separates it by its channels and then uses min-max scaling on the values
# within each channel independently to scale between 0 and 1
def scale_image(image):
    image_min = tf.reduce_min(image)
    image_max = tf.reduce_max(image)
    image_range = image_max-image_min
    return tf.divide(tf.subtract(image, image_min), image_range)

def preprocess(image):
    assert(tf.reduce_max(image) <= 1.001)
    imagenet_mean = [0.40760392,  0.45795686,  0.48501961]
    image_torch = 255 * tf.transpose((image[:,:,::-1] - imagenet_mean), [2,0,1])
    return image_torch

def deprocess(image):
    imagenet_mean = [0.40760392,  0.45795686,  0.48501961]
    image = (tf.transpose(image, [1,2,0])/255. + imagenet_mean)[:,:,::-1]
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Gradient should be the same shape as input images (after being resized if needed)
# Mask should be a 2D tensor with same width and height as gradient but only one channel and contain only 1's and 0's
def mask_grad(grad, mask):
    channels = tf.split(grad, 3, axis=2)
    for i in range(len(channels)):
        channels[i] = tf.expand_dims(tf.multiply(channels[i][:,:,0],mask), 2)
    return tf.concat(channels, 2)
