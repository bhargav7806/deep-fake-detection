import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Resizing , Rescaling , InputLayer , Conv2D , BatchNormalization , MaxPool2D , Flatten , Dropout , Dense , GlobalAveragePooling2D , Input
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy , F1Score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix , roc_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from utils import process_image


train_directory = r"D:\AI\projects\deep fake detection\data\Data\train"
test_directory = r"D:\AI\projects\deep fake detection\data\Data\test"
val_directory = r"D:\AI\projects\deep fake detection\data\Data\val"

CONFIGURATION = {
    'BATCH_SIZE' : 32 ,
    'IM_SIZE' : 256 ,
    'LEARNING_RATE' : 0.0001,
    'REGULARIZATION_RATE' : 0.001 ,
    'DROPOUT_RATE' : 0.3 ,
    'KERNEL_SIZE' : 3 ,
    'N_FILTERS' : 6 ,
    'N_EPOCHS' : 25 ,
    'POOL_SIZE' : 2 ,
    'N_STRIDES' : 1 ,
    'DENSE1' : 128 ,
    'DENSE2' : 64 ,
    'DENSE3' : 32 ,
    'NUM_CLASSES' : 2

}
class_names = ['AI' , 'REAL']  # real - 1 , ai - 0

# data processing

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels='inferred',
    label_mode='binary',
    class_names=class_names,
    color_mode='rgb',
    batch_size=CONFIGURATION['BATCH_SIZE'],
    image_size=(CONFIGURATION['IM_SIZE'] , CONFIGURATION['IM_SIZE']),
    shuffle=True,
    seed=43,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_directory,
    labels='inferred',
    label_mode='binary',
    class_names=class_names,
    color_mode='rgb',
    batch_size=CONFIGURATION['BATCH_SIZE'],
    image_size=(CONFIGURATION['IM_SIZE'] , CONFIGURATION['IM_SIZE']),
    shuffle=False,
    seed=43,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    labels='inferred',
    label_mode='binary',
    class_names=class_names,
    color_mode='rgb',
    batch_size=CONFIGURATION['BATCH_SIZE'],
    image_size=(CONFIGURATION['IM_SIZE'] , CONFIGURATION['IM_SIZE']),
    shuffle=False,
    seed=43,
)

resize_rescale_layer = Sequential([
    Resizing(CONFIGURATION['IM_SIZE'] , CONFIGURATION['IM_SIZE']),
    Rescaling(1./255.)
])

training_dataset = (
    train_ds
    .prefetch(tf.data.AUTOTUNE)
)
validation_dataset =(
    val_ds
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset =(
    test_ds
    .prefetch(tf.data.AUTOTUNE)
)

## Model training 

backbone = ResNet50 (
    include_top = False ,
    weights = 'imagenet' ,
    input_shape = (CONFIGURATION['IM_SIZE'] , CONFIGURATION['IM_SIZE'] , 3)
)

backbone.trainable = False # freeze the model

modelcheckpointcallback = ModelCheckpoint(
    filepath = "resnet_model.keras",
    monitor = "val_loss",
    save_best_only = True,
    mode = 'min',
    verbose = 1
)

input_tensor = Input(shape = (CONFIGURATION['IM_SIZE'] , CONFIGURATION['IM_SIZE'] , 3))


x = backbone(input_tensor , training = False)  # here trainable false keeps batchnorm inference
x = GlobalAveragePooling2D()(x)
x = Dense(CONFIGURATION['DENSE1'] , activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dense(CONFIGURATION['DENSE2'] , activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dense(CONFIGURATION['DENSE3'] , activation = 'relu')(x)
x = BatchNormalization()(x)
final_output_tensor = Dense(1 , activation = 'sigmoid')(x)

fine_tuned = tf.keras.Model(inputs=input_tensor, outputs=final_output_tensor)

fine_tuned.compile(loss = BinaryCrossentropy() , optimizer = Adam(0.00001) , metrics = [BinaryAccuracy()])

history_fine_tuned = fine_tuned.fit(training_dataset , validation_data = validation_dataset , epochs = 15)

fine_tuned.trainable = True

for layer in backbone.layers[:-20]:
    layer.trainable = False

fine_tuned.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine_tuned = fine_tuned.fit(training_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks = [modelcheckpointcallback]
)


fine_tuned_model = tf.keras.models.load_model("D:\AI\projects\deep fake detection")
with open("D:\AI\projects\deep fake detection" , "rb") as file:
  history_fine_tuned = pickle.load(file)

plt.plot(history_fine_tuned['loss'])
plt.plot(history_fine_tuned['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.plot(history_fine_tuned['accuracy'])
plt.plot(history_fine_tuned['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')