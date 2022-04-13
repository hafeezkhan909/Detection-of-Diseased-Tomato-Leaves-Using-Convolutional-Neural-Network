import numpy as np
import detection
from keras.preprocessing import image
test_image = image.load_img('dataset\ce1dd162-881b-4097-89b4-eccee085fbfa___Matt.S_CG 0732.jpg', target_size = (256,256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'non_healthy_tomato'
else:
  prediction = 'healthy_tomato'
print(prediction)
