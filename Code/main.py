## if the processed files do not exist then import data_maker, otherwise comment it out and move to the code below
#import data_maker
#data_maker.make_data('raw data\\3 Comp\\three_comp_sams_1k.mat', 0)
#data_maker.make_data('raw data\\3 Comp\\three_comp_sams_5k.mat', 0)



import multi_class_model

multi_class_model.train(0.7)
#multi_class_model.load_and_test_model()
#multi_class_model.load_and_test_model('trained models\gold_odt_methoxy multi class\my_checkpoint')
##########################################
# training a model
#binary_class_model.train()
# testing out trained models
#train_model.load_and_test_model('trained models\odt_methoxy binary class\latest\my_checkpoint')
###########################################

import numpy as np
import matplotlib.pyplot as plt
import data_maker

#data_maker.make_data('/home/shah/PycharmProjects/NNs for Chemistry/raw data/5 Comp/five_sames_binned.mat',0)
# testing out new image
#fiveSams1k = np.load('/home/shah/PycharmProjects/NNs for Chemistry/Results/fiveSams1k-new_labels.npy')
new_test_data = np.load(
    '/home/shah/PycharmProjects/NNs for Chemistry/processed data/New multi sams data-stamped_image_1k_4x4.npy')
print(new_test_data.shape)
new_labels = multi_class_model.load_and_test_model(new_test_data)
final_labels = []
for a in range(256):
  final_labels.append(new_labels[a * 256:a * 256 + 256, :])

image = np.array(final_labels)
#np.save('/home/shah/PycharmProjects/NNs for Chemistry/Results/fiveSams1k-image', image)
print('shape of image is ', np.shape(image))
#image = np.load('/home/shah/PycharmProjects/NNs for Chemistry/Results/old_resutls/5 sams 1k shot image and 1k shot training/fiveSams1k-image.npy')

img = image[:,:,0]
# print(np.shape(img))
#
# for x in range(1024):
#     for y in range(1024):
#         if image[x,y,4]<0.1:
#             img[x,y] = 0.00000000
#
#         else:
#             img[x,y] = 1.0000000

plt.figure(figsize=[10, 10])
plt.imshow(img)
plt.colorbar()
plt.show()

# plt.hist(img.flatten(), bins=[-.5,.5,1.5], ec="k")
# plt.xticks((0,1))
# plt.show()


plt.figure(figsize=[10,10])
plt.imshow(image[:,:,1])
plt.colorbar()
plt.show()


plt.figure(figsize=[10,10])
plt.imshow(image[:,:,2])
plt.colorbar()
plt.show()

plt.figure(figsize=[10,10])
plt.imshow(image[:,:,3])
plt.colorbar()
plt.show()

plt.figure(figsize=[10,10])
plt.imshow(image[:,:,4])
plt.colorbar()
plt.show()
#def random_one_hot_labels(shape):
#  n, n_class = shape
#  print(n, n_class)
#  classes = np.random.randint(0, n_class, n)
#  print('classes', classes)
#  labels = np.zeros((n, n_class))
#  print('labels', labels)
#  labels[np.arange(n), classes] = 1
#  print('labels', labels)
#  return labels
#random_one_hot_labels([2,3])
