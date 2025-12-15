import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128,128)
batch_size = 16

train_datagen=ImageDataGenerator(rescale=1./255)
val_datagen=ImageDataGenerator(rescale=1./255) 

train_data=train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='categorical'

)

val_data=train_datagen.flow_from_directory(
    'data/val',
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='categorical'

)
num_class = len(train_data.class_indices)

model=models.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3),padding="same"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation="relu",padding="same"),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(254,(3,3),activation="relu",padding="same"),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128,(3,3),activation="relu",padding="same"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dropout(.3),
    layers.Dense(num_class,activation="softmax")
]
)

model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)

model.fit(
    train_data,
    epochs = 10,
    validation_data = val_data
)

model.save("mybatch_students_cnn_classifier.h5")

print(model.summary())
