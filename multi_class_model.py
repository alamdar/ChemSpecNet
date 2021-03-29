import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import data_wrangling
import os



def create_model():
  model = tf.keras.models.Sequential([
#    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(71, activation=tf.nn.relu),
#    tf.keras.layers.Dropout(0.25),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Conv1D(filters=71, kernel_size=10),
    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax),
#    tf.keras.layers.Dropout(0.25)
  ])

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('acc') > 0.98):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True


def train(training_fraction):
  # Include the epoch in the file name (uses `str.format`)
  checkpoint_path = "/home/shah/PycharmProjects/NNs for Chemistry/trained models/multi_sams_cp-{epoch:04d}.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights every 5 epochs
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

  # Create a new model instance
  model = create_model()


  [training_spectra, training_labels, test_spectra, test_labels] = \
    data_wrangling.generate_five_sams_data(training_fraction)#data_wrangling.generate_five_sams_binned_data(training_fraction)
  callbacks = myCallback()

  #history = model.fit(training_spectra, training_labels, epochs=10, callbacks=[cp_callback], validation_data=[test_spectra,test_labels], use_multiprocessing=True)
  history = model.fit(training_spectra, training_labels, epochs=15, callbacks=[cp_callback], use_multiprocessing=True)
  model.summary()
  validation_loss = model.evaluate(test_spectra, test_labels)
  model.save_weights('/home/shah/PycharmProjects/NNs for Chemistry/trained models/my5component_checkpoint')
  print(model.predict(test_spectra))

  #acc = history.history['acc']
  #val_acc = history.history['val_acc']
  #loss = history.history['loss']
  #val_loss = history.history['val_loss']

  #epochs = range(len(acc))
  #plt.figure()
  #plt.plot(epochs, acc, 'r', label='Training accuracy')
  #plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  #plt.title('Training and validation accuracy')
  #plt.legend(loc=0)
  #plt.show()
  ## testing out new image
  fiveSams1k = np.load('/home/shah/PycharmProjects/NNs for Chemistry/processed data/multi sams data-stamped/fiveSams1k.npy')
  ##fiveSams5k = np.load('processed data\multi sams data-stamped\\fiveSams5k.npy')
  #threeSams1k = np.load('processed data\\3 comp\sams_3comp_1k.npy')
  new_labels = model.predict(fiveSams1k)
  print(np.shape(new_labels))
  np.save('/home/shah/PycharmProjects/NNs for Chemistry/Results/fiveSams1k-new_labels', new_labels)
  #final_labels = []
  #for a in range(1024):
  #  final_labels.append(new_labels[a * 1024:a * 1024 + 1024, :])

  #image = np.array(final_labels)
  #np.save('Results\\3sams1k-image', image)
  #print('shape of image is ', np.shape(image))


  #plt.figimage(image)
  #plt.show()





def load_and_test_model(new_test_data):
  #odt_methoxy = np.load('processed data\gold_odt_methoxy\processed_odt_methoxy.npy')
  #fiveSams1k = np.load(
  #  '/home/shah/PycharmProjects/NNs for Chemistry/processed data/multi sams data-stamped/fiveSams1k.npy')
  model = create_model()
  model.load_weights('/home/shah/PycharmProjects/NNs for Chemistry/trained models/my5component_checkpoint')
  new_labels = model.predict(new_test_data)
  print(np.shape(new_labels))
  #np.save('/home/shah/PycharmProjects/NNs for Chemistry/Results/fiveSams1k-new_labels', new_labels)
  return new_labels


def eval_each_class(single_sams_test_spec, single_sams_test_labels):
  model = create_model()
  model.load_weights('/home/shah/PycharmProjects/NNs for Chemistry/trained models/my5component_checkpoint')
  model.evaluate(single_sams_test_spec, single_sams_test_labels)
  return None
