import tensorflow as tf
import util

# NST model to extract style and content from provided image
# Parameters:
# style_layers - list of strings with names of layers to extract style features from
# content_layers - list of strings with names of layers to extract content features from
# model - pretrained CNN model
class NSTModel(tf.keras.models.Model):    
    def __init__(self, style_layers, content_layers, style_weight, content_weight, model, image_dim):
        super(NSTModel, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.image_dim = image_dim
        outputs = [model.get_layer(name).output for name in style_layers+content_layers]
        model = tf.keras.Model([model.input], outputs)
        self.model = model
        self.model.trainable = False

    # Takes an image as input, extracts style and content features
    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = tf.keras.applications.vgg19.preprocess_input(inputs*255)
        preprocessed_input = tf.reshape(inputs, (1, self.image_dim[0], self.image_dim[1], self.image_dim[2]))
        outputs = self.model(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])
        
        content_outputs = [tf.reshape(output, (output.shape[1]*output.shape[2], output.shape[3]))
                        for output in content_outputs]

        style_outputs = [util.gram_matrix(style_output, shift=True)
                        for style_output in style_outputs]

        content_dict = dict(zip(self.content_layers, content_outputs))
        style_dict = dict(zip(self.style_layers, style_outputs))

        return {'content': content_dict, 'style': style_dict}

    ### Loss functions

    # Takes content and style targets along with dictionary with content/style features of generated iamge and returns scalar loss
    def total_loss(self, content_targets, style_targets, generated):
        l = self.content_weight*self.content_loss(content_targets, generated) + self.style_weight*self.style_loss(style_targets, generated)
        return l

    def content_loss(self, content_targets_dict, generated):
        generated = generated['content']
        l = tf.add_n([0.5*tf.reduce_sum(tf.square(tf.subtract(content_targets_dict[name], generated[name])))]
                        for name in content_targets_dict.keys())
        l = l / self.num_content_layers
        return l

    def style_loss(self, style_targets_dict, generated):
        generated = generated['style']
        l = tf.add_n([tf.reduce_mean(tf.square(tf.subtract(style_targets_dict[name], generated[name])))]
                        for name in style_targets_dict.keys())
        l = l / self.num_style_layers
        return l

    