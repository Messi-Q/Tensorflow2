import tensorflow as tf
import numpy as np

print(tf.__version__)
print(tf.keras.__version__)

############## 1. keras.Model API ###############
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu(inputs))
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


############## 2. Inherit Model ###############
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


############## 3. Sequential ###############
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(50, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))


############## 4. Model Reuse ###############
vgg16 = tf.keras.applications.VGG16()
feature_list = [layer.output for layer in vgg16.layers]
feat_ext_model = tf.keras.Model(inputs=vgg16.input, outputs=feature_list)
img = np.random.random((1, 224, 224, 3).astype('float32'))
ext_features = feat_ext_model(img)


############## 5. Multiple input and output ###############
image_input = tf.keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = tf.keras.Input(shape=(None, 10), name='ts_input')
x1 = tf.keras.layers.Conv2D(3, 3)(image_input)
x1 = tf.keras.layers.GlobalMaxPooling2D()(x1)
x2 = tf.keras.layers.Conv1D(3, 3)(timeseries_input)
x2 = tf.keras.layers.GlobalMaxPooling1D()(x2)
x = tf.keras.layers.concatenate([x1, x2])
score_output = tf.keras.layers.Dense(1, name='score_output')(x)
class_output = tf.keras.layers.Dense(5, activation='softmax', name='class_output')(x)
model = tf.keras.keras.Model(inputs=[image_input, timeseries_input],
                             outputs=[score_output, class_output])
tf.keras.utils.plot_model(model, 'multi_input_output_model.png', show_shapes=True)
