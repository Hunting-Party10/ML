import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test,y_test) = mnist.load_data()

prediction_model = tf.keras.models.load_model('mnist_reader.h5')

prediction_model.summary()

predictions = prediction_model.predict(x_test)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()