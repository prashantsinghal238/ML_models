from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import numpy as np
import os
#path of dataset
path=r"C:\Users\psing\Desktop\projects\finalyearproject\plants_disease_detection\Potato"  
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
def train_data(path): 
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
    training_set = train_datagen.flow_from_directory(os.path.join(path,"Train"),
                                                    target_size = (224, 224),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')
    return training_set
def validation_data(path):                                                     
    test_datagen = ImageDataGenerator(rescale = 1./255) 
    test_set = test_datagen.flow_from_directory(os.path.join(path,"Valid"),
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical')     
    return test_set  
def trainupper(model,top=4):        
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    tot_layers=len(model.layers)
    for layer in model.layers[:tot_layers-top]:
        layer.trainable = False
    for layer in model.layers[tot_layers-top:]:
        layer.trainable = True
def save(model,str(name)="model"):
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(name+".h5")
    return 1           
trainupper(model,10)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics = ['accuracy'])  
model.fit(train_data(path),
                         steps_per_epoch = 30,
                         epochs = 10,
                         validation_data = validation_data(path),
                         validation_steps = 2)
#model.summary()                         
success=save(model,name)                           