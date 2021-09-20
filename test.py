# Import libraries
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import cv2
import pickle
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# load image -> face detection
sample_img = cv2.imread('sample/2.jpeg')

results = detector.detect_faces(sample_img)

x, y, width, height = results[0]['box']

face = sample_img[y:y + height, x:x + width]

# cv2.imshow('output', face)
# cv2.waitKey(0)

#  extract it's features

image = Image.fromarray(face)
image = image.resize((224, 224))

face_array = np.asarray(image)

face_array = face_array.astype('float32')

expanded_image = np.expand_dims(face_array, axis=0)

preprocessed_image = preprocess_input(expanded_image)

result = model.predict(preprocessed_image).flatten()
# print(result)
# print(result.shape)

# find the cosine distance of current image with all the 8655 features
similarity = []

for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])


# sort on bases on second index similarity score (key=lambda x:x[1])
index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][0]

temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)

# recommend that image