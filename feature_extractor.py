# Import libraries

import pickle

import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl', 'rb'))

# Create model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

print(model.summary())


def feature_extractor(img_path, model):
    # load the image
    img = image.load_img(img_path, target_size=(224, 244))

    # convert image to array
    img_array = image.img_to_array(img)

    # reshape or expand dimension
    expanded_img = np.expand_dims(img_array, axis=0)

    # preprocess image
    preprocessed_img = preprocess_input(expanded_img)

    # predict : model will return 2048 features for each image
    results = model.predict(preprocessed_img).flatten()

    return results


features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

#pickle.dump(features, open('embedding.pkl', 'wb'))
