import numpy as np
import tensorflow as tf
import data_wrangling
import matplotlib.pyplot as plt


def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(71, activation=tf.nn.relu),
    tf.keras.layers.Dense(500, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])
  model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
  return model


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('acc') > 0.98):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True

def train(training_fraction):
  [training_spectra, training_labels, test_spectra, test_labels] = data_wrangling.generate_odt_methoxy_binary_data(training_fraction)
  callbacks = myCallback()
  model = create_model()
  model.fit(training_spectra, training_labels, epochs=5, callbacks=[callbacks])
  test_loss = model.evaluate(test_spectra, test_labels)
  model.save_weights('trained models\odt_methoxy binary class\latest\my_checkpoint')
  return 0

def load_and_test_model(saved_model_path):
  model = create_model()
  model.load_weights(saved_model_path)
  odt_methoxy = np.load('processed data\gold_odt_methoxy\processed_odt_methoxy.npy')
  new_labels = model.predict(odt_methoxy)
  print(np.shape(new_labels))
  print(np.round(new_labels[:10]))
  final_labels = []
  for a in range(1024):
    final_labels.append(new_labels[a * 1024:a * 1024 + 1024])

  print('shape of labels is ', np.shape(final_labels))
  image = np.array(final_labels)

  print('shape of image is ', np.shape(image[:, :, 0]))

  plt.figimage(np.round(image[:, :, 0]))
  plt.show()
  return 0
