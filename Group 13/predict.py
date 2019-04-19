from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model

HEIGHT = 150
WIDTH = 150
model_weights_path="checkpoints/vgg16_model_weights.h5"
model=VGG16(weights='imagenet',include_top=False,input_shape=(HEIGHT, WIDTH, 3))
x=model.output
dropout=0.5
num_classes=24
fc_layers = [1024, 1024]
x= Flatten()(x)
for fc in fc_layers:
        # New FC layer, random init
    x = Dense(fc, activation='relu')(x) 
    x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    model= Model(inputs=model.input, outputs=predictions)

#print(model.summary())
model.load_weights(model_weights_path)

file="predict.jpg"
img_width=150
img_height=150
import numpy as np
x = load_img(file, target_size=(img_width,img_height))
x = img_to_array(x)  
x = np.expand_dims(x, axis=0)
array = model.predict(x)
#print(array)
answer = np.argmax(array)
#print(answer)
class_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
print(class_list[answer])

