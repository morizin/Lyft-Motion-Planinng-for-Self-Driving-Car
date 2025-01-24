import tensorflow as tf
import numpy as np

def modified_resnet50():
    # Model with 3 input channel dim with pretrained weights
    pretrained_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    
    # Model with 25 input channel dim without pretrained weights
    modified_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(None, None, 25))
    
    for pretrained_model_layer, modified_model_layer in zip(pretrained_model.layers, modified_model.layers):
        layer_to_modify = ['conv1_conv']  # conv1_conv is the name of the layer that takes the input and will be modified
        if pretrained_model_layer.name in layer_to_modify:
            kernel = pretrained_model_layer.get_weights()[0]  # kernel weight shape is (7, 7, 3, 64)
            bias = pretrained_model_layer.get_weights()[1]
            
            # Concatenating along channel axis to make channel dimension 25
            weights = np.concatenate((kernel[:, :, -1:, :], np.tile(kernel, [1, 1, 8, 1])), axis=-2)
            modified_model_layer.set_weights((weights, bias))
        else:
            modified_model_layer.set_weights(pretrained_model_layer.get_weights())
    
    return modified_model

class LyftModel(tf.keras.Model):
    def __init__(self, num_modes=3, future_pred_frames=50):
        super(LyftModel, self).__init__()
        
        self.model = modified_resnet50()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.2)
        
        self.future_len = num_modes * future_pred_frames * 2
        self.future_pred_frames = future_pred_frames
        self.num_modes = num_modes
        
        self.dense1 = tf.keras.layers.Dense(self.future_len + self.num_modes)
    
    def call(self, inputs):
        x = self.model(inputs)
        x = self.gap(x)
        x = self.dropout(x)
        x = self.dense1(x)
        
        batch_size, _ = x.shape
        pred, confidence = tf.split(x, num_or_size_splits=[self.future_len, self.num_modes], axis=1)
        assert confidence.shape == (batch_size, self.num_modes), f'confidence got shape {confidence.shape}'
        pred = tf.reshape(pred, shape=(batch_size, self.num_modes, self.future_pred_frames, 2))
        confidence = tf.nn.softmax(confidence, axis=1)
        return pred, confidence