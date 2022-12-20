import tensorflow as tf
from tensorflow.keras.applications import VGG19
import numpy as np
import util
import upscale
from NSTModel import NSTModel
import time
import argparse
import cv2

def require_args(parser):
    content_layer_choices = ["block2_conv2", "block3_conv2", "block4_conv2", "block5_conv2"]
    style_layer_choices = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1',
                            'block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2', 'block5_conv2',
                            'block3_conv3', 'block4_conv3', 'block5_conv3',
                            'block3_conv4', 'block4_conv4', 'block5_conv4']
    parser.add_argument("-preserve_color", action='store_true', default=False, dest='color_preserve',
                        help="use to preserve the original color of the content image (by default it will be overriden by the style image's color)")
    parser.add_argument("-style_image_path", required=True,
                        help="file path to the desired style image")
    parser.add_argument("-content_image_path", required=True,
                        help="file path to the desired content image")
    parser.add_argument("-output_image_path", required=True,
                        help="file path where the generated image will go")
    parser.add_argument("-style_content_ratio", default=0.001, dest='ratio', type=float,
                        help="Equal to alpha/beta (content/style) in paper - larger number emphasizes content more (default is 0.001)")
    parser.add_argument("-variation_weight", default=0, dest='total_variation_weight', type=float,
                        help="How much to penalize noise in the output image (default is 0.0001)")
    parser.add_argument("-min_improve", default=100, type=int,
                        help="Minimum decreases in loss to continue optimizing (default is 100)")
    parser.add_argument("-max_epochs", default=100, type=int,
                        help="Maximum number of epochs to allow the program to run (default is 100)")
    parser.add_argument("-content_layer", default='block5_conv2', nargs="*", choices=content_layer_choices,
                        help="Layer(s) to use for content features (default is block5_conv2)")
    parser.add_argument("-style_layer", default=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'], 
                        nargs="*", choices=style_layer_choices,
                        help="Layer(s) to use for style features (default is all blocks conv1)")
    parser.add_argument("-pooling", default='avg', choices=['avg', 'max'],
                        help="Whether to use avg or max pooling (default is avg)")   
    parser.add_argument("-start_image", default='content', choices=['content', 'style', 'white_noise'],
                        help="What image to use as the starting point for the algorithm (default is content)")   
    parser.add_argument("-max_size", default=500, type=int,
                        help="Maximum size for input images (also determines the output image's dimensions)")
    parser.add_argument("-mask_path", default=None,
                        help="Optional path to a mask file that will determine what part(s) of the image to apply style to") 
    parser.add_argument("-invert_mask", action='store_true', default=False,
                        help="Whether to transfer style to whitespace (if selected) or only to non-whitespace (the default)") 
    parser.add_argument("-upscale", action='store_true', default=False,
                        help="Whether to automatically upscale the output image")               
    return vars(parser.parse_args())

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = extractor.total_loss(content_targets, style_targets, outputs)
        loss += total_variation_weight*tf.image.total_variation(image)
        grad = tf.clip_by_norm(tape.gradient(loss, image), 255)
        opt.apply_gradients([(grad, image)])
    #image.assign(util.scale_image(image))
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
    return(loss)

def optimize(image):
    # Tracking stats
    start = time.time()
    loss_list = []
    diff_list = []
    step = 0

    # Main optimization loop
    while ((sum(diff_list[-leeway_steps:])/leeway_steps) < -min_improve or len(loss_list) < min_epochs) and step <= max_steps:
        step += 1
        l = train_step(image)
        print(".", end='', flush=True)
        if step % steps_per_epoch == 0:
            loss_list.append(l)
            loss_diff = l - loss_list[len(loss_list)-2]
            diff_list.append(loss_diff)
            print("Train step: %d   Total Loss %d   Change in Loss %d"%(step, l, loss_diff))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    ### Reconstructing final image
    # Applying mask if necessary
    if args['mask_path']:
        if args['invert_mask']:
            image = tf.where(mask_image==255, image, content_image)
        else:
            image = tf.where(mask_image!=255, image, content_image)

    # Converting to BGR color space to save image
    r, g, b = cv2.split(np.array(image*255, dtype=np.uint8))
    bgr_stylized = cv2.merge((b, g, r))
    return bgr_stylized

if __name__ == "__main__":
    args = require_args(argparse.ArgumentParser(
        description='Program that uses gradient descent to generate a new image that attempts to jointly optimize ' +
         'similarilty to a provide style image\'s style and a content image\'s content'))

    # Image paths
    style_image_path = args['style_image_path']
    content_image_path = args['content_image_path']
    if args['mask_path']:
        mask_path = args['mask_path']
        # Transform mask to only trues and falses
        # White areas become false unless inverting
        # False means that the style will not be transferred to that section of the picture
        b, g, r = cv2.split(cv2.imread(mask_path))
        mask_image = cv2.merge((r, g, b))

    # RGB Images
    b, g, r = cv2.split(cv2.imread(style_image_path))
    style_image = cv2.merge((r, g, b))
    b, g, r = cv2.split(cv2.imread(content_image_path))
    content_image = cv2.merge((r, g, b))

    # Rescaling content image dimensions if it's too large for the program to run
    max_size = args['max_size'] # Largest number of pixels for longest side of image
    largest_dim = np.argmax(content_image.shape)
    if content_image.shape[largest_dim] > max_size:
        # Rescale content image
        curr_size = content_image.shape[largest_dim]
        scale_ratio = curr_size / max_size
        new_dims = [int(content_image.shape[0] / scale_ratio), int(content_image.shape[1] / scale_ratio)]
        content_image = np.array(tf.image.resize(content_image, new_dims, preserve_aspect_ratio=True), dtype=np.uint8)
        
        if args['mask_path']:
            # Rescale mask image
            mask_image = tf.image.resize(mask_image, new_dims, preserve_aspect_ratio=True)#[:,:,0]

    # Adjusting size of style image to match that of the content image
    content_image_dimensions = content_image.shape
    style_image = np.array(tf.image.resize(style_image, content_image_dimensions[:2]), dtype=np.uint8)

    # Preprocessing input images
    white_noise = np.random.uniform(size=content_image_dimensions)
    style_image = util.scale_image(tf.Variable(tf.cast(tf.convert_to_tensor(style_image),
                                                       dtype=tf.float32), trainable=True))
    content_image = util.scale_image(tf.Variable(tf.cast(tf.convert_to_tensor(content_image),
                                                         dtype=tf.float32), trainable=True))
    image = tf.Variable(tf.cast(tf.convert_to_tensor(white_noise), dtype=tf.float32), trainable=True)

    # Select starting image
    if args['start_image'] == 'content':
        image = tf.Variable(tf.cast(tf.convert_to_tensor(content_image), dtype=tf.float32), trainable=True)
    elif args['start_image'] == 'style':
        image = tf.Variable(tf.cast(tf.convert_to_tensor(style_image), dtype=tf.float32), trainable=True)        

    # Which layers to use
    c_layers = args['content_layer']
    if type(c_layers) != list:
        c_layers = [c_layers]
    s_layers = args['style_layer']
    if type(s_layers) != list:
        s_layers = [s_layers]
    for layer in c_layers:
        if layer in s_layers:
            s_layers.remove(layer)

    # How much to weight style, content, and variation
    ratio = args['ratio']
    content_weight = 20
    style_weight = content_weight / ratio
    total_variation_weight = args['total_variation_weight']

    # How long to keep optimizing
    steps_per_epoch = 100
    min_improve = args['min_improve']
    leeway_steps = 5
    min_epochs = 5
    max_epochs = args['max_epochs']
    max_steps = steps_per_epoch * max_epochs

    # Creating model and optimizer
    model = VGG19(include_top = False, pooling = args['pooling'], input_shape = content_image_dimensions)
    model.load_weights('vgg19_norm_weights.h5')
    extractor = NSTModel(s_layers, c_layers, style_weight, content_weight, model, content_image_dimensions)
    opt = tf.optimizers.Adam(learning_rate=0.001, epsilon=0.1)

    # Passing style and content images through network to get target features
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # Creating new image
    output_image = optimize(image)

    # Applying color adjustment if necessary
    if args['color_preserve']:
        image = util.scale_image(tf.Variable(tf.cast(tf.convert_to_tensor(output_image),
                                                         dtype=tf.float32), trainable=True))
        # Extract luminance channel from stylized image
        yiq_image = tf.image.rgb_to_yiq(image)
        y_image = tf.split(yiq_image, yiq_image.shape[2], 2)[0]

        # Extract iq channels from original content image
        yiq_content = tf.image.rgb_to_yiq(content_image)
        iq_content = tf.split(yiq_content, yiq_content.shape[2], 2)[1:]

        # Combine luminance from stylized image with color from content
        yiq_new = tf.concat([y_image, iq_content[0], iq_content[1]], axis=2)
        rgb_new = tf.image.yiq_to_rgb(yiq_new)

        # Save newly colored image
        rgb_new = np.array(rgb_new*255, dtype=np.uint8)
        r, g, b = cv2.split(rgb_new)
        output_image_original_color =  cv2.merge((b, g, r))

    # Save stylized image along with additional recolored/upscaled version(s) if necessary
    cv2.imwrite(args['output_image_path'], output_image)
    if args['upscale']:
        upscale.upscale_image(args['output_image_path'])
    if args['color_preserve']:
        new_path = args['output_image_path'][:-4] # remove '.jpg'
        new_path = new_path + " recolored.jpg"
        cv2.imwrite(new_path, output_image_original_color)
        if args['upscale']:
            upscale.upscale_image(new_path)
