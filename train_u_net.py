"""
This script trains the model
"""
import pathlib
import tensorflow as tf
from model import build_unet_model
from helper_functions import dice_score, plot_loss_curves
from custom_dataset import train_data, test_data

# Instantiate model
model_0 = build_unet_model(input_shape=(128, 128, 3))

# Compile model
model_0.compile(optimizer=tf.keras.optimizers.SGD(),
                loss=dice_score,
                metrics=['accuracy'])

# Setup hyperparameter
epochs=5
batch_size = 1

# Train the model
model_history = model_0.fit_generator(train_data,
                                      epochs=5,
                                      steps_per_epoch=len(train_data),
                                      validation_data=test_data,
                                      validation_steps=len(test_data))

plot_loss_curves(model_history)

model_0.evaluate(test_data)

save_path = 'ships_prediction_model'
model_0.save(save_path)
