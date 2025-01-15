import tensorflow as tf
from preprocessing import test_generator

#saved model
model = tf.keras.models.load_model('mobilenet_food_classification.h5')

#evaluate model on test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

#evaluation results
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')