import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
data = load_digits()
iris_target1 = data.target
iris_data = np.float32(data.data)
iris_target = np.float32(tf.keras.utils.to_categorical(iris_target1, num_classes=10))
iris_data = tf.data.Dataset.from_tensor_slices(iris_data).batch(50)
iris_target = tf.data.Dataset.from_tensor_slices(iris_target).batch(50)
inputs = tf.keras.layers.Input(shape=(64))
print(len(load_digits().data[0]))
print(len(load_digits().target_names))
x = tf.keras.layers.Dense(32, activation='relu')(inputs)
y = tf.keras.layers.Dense(64, activation='relu')(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
opt = tf.optimizers.Adam(1e-3)
for epoch in range(500):
    for _data, lable in zip(iris_data, iris_target):
        with tf.GradientTape() as tape:
            logits = model(_data)
            loss_value = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=lable, y_pred=logits))
            grads = tape.gradient(loss_value, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
print(loss_value.numpy())
model.save('load.digits.h5')
new_model = tf.keras.models.load_model('load.digits.h5')
new_prediction = new_model.predict(iris_data)
new_target = tf.argmax(new_prediction, axis=-1).numpy()
print(new_target)
print(iris_target1)
print((new_target == iris_target1).all())
