import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2  # Using a faster model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocessing import train_generator, validation_generator

#set parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32
NUM_CLASSES = 11

#build model - mobileNetV2
def create_mobilenet_model():
    base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                             include_top=False,
                             weights='imagenet')
    #freeze the bl model
    base_model.trainable = False
    #add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

#create the model
model = create_mobilenet_model()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# callback/earlystopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)
#epoch
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]
)
#save
model.save('mobilenet_food_classification.h5')