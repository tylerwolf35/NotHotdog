from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from hotdog import identify


def predict(image1):
    model = VGG16()
    image = load_img(image1, target_size=(224, 224))
    # pixels -> numpy array
    image = img_to_array(image)
    # reshape data
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare for VGG model
    image = preprocess_input(image)
    # predict
    yhat = model.predict(image)
    # convert
    label = decode_predictions(yhat)
    # retrieve result (hotdog or not hotdog)
    label = label[0][0]
    return label
    identify(label)
