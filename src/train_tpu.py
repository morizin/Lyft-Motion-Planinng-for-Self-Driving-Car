
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from kaggle_datasets import KaggleDatasets
from collections import namedtuple
import matplotlib.pyplot as plt
import time
from utils import set_seed, get_device, get_strategy

import tensorflow as tf
import time
from collections import namedtuple
from models_tf import LyftModel, modified_resnet50

def get_model_and_variables(global_batch_size, img_dim, channel_dim):
    model = LyftModel()
    model.build((global_batch_size, img_dim, img_dim, channel_dim))
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Loss function
    loss_func = lambda a, b, c, d: tf.nn.compute_average_loss(neg_multi_log_likelihood(a, b, c, d), global_batch_size=global_batch_size)
    
    # Transformation function
    transf_points = lambda pred, world_from_agent: transform_points(pred, world_from_agent)
    
    # Metrics
    training_loss = tf.keras.metrics.Sum()
    validation_loss = tf.keras.metrics.Sum()
    
    return model, optimizer, loss_func, transf_points, training_loss, validation_loss

def train_model(train_dataset, valid_dataset, model, optimizer, loss_func, transf_points, training_loss, validation_loss, epochs=5, steps_per_epoch=200, validation_step=100):
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)
    
    start_time = epoch_start_time = time.time()
    
    print("Steps per epoch:", steps_per_epoch)
    History = namedtuple('History', 'history')
    history = History(history={'train_loss': [], 'val_loss': []})
    
    epoch = 0
    for step, (images, target_pos, target_avails) in enumerate(train_dist_dataset):
        strategy.run(train_step, args=(images, target_pos, target_avails))
        
        if ((step + 1) // steps_per_epoch) > epoch:
            for val_step, (images, target_pos, target_avails, world_from_agent) in enumerate(valid_dist_dataset):
                strategy.run(valid_step, args=(images, target_pos, target_avails, world_from_agent))
                if (val_step + 1) % validation_step == 0:
                    break
            
            history.history['train_loss'].append(training_loss.result().numpy() / steps_per_epoch)
            history.history['val_loss'].append((validation_loss.result().numpy() / validation_step))
            
            epoch_time = time.time() - epoch_start_time
            print(f'\nEPOCH {epoch + 1}/{epochs}')
            print(f'time: {epoch_time:.1f}s loss: {history.history["train_loss"][-1]:.4f} val_loss: {history.history["val_loss"][-1]:.4f}')
            
            model.save_weights(f'./epoch {epoch + 1}, train_loss {history.history["train_loss"][-1]:.4f}, val_loss {history.history["val_loss"][-1]:.4f} model.h5')
            
            epoch = (step + 1) // steps_per_epoch
            epoch_start_time = time.time()
            validation_loss.reset_states()
            training_loss.reset_states()
        
        if epoch >= epochs:
            break

@tf.function
def train_step(image, target_pos, target_avail):
    with tf.GradientTape() as tape:
        pred, confidence = model(image, training=True)
        loss_value = loss_func(target_pos, pred, confidence, target_avail)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    training_loss.update_state(loss_value)

@tf.function
def valid_step(image, target_pos, target_avail, world_from_agent):
    pred, confidence = model(image, training=False)
    pred = transf_points(pred, world_from_agent)
    val_loss = loss_func(target_pos, pred, confidence, target_avail)
    validation_loss.update_state(val_loss)


if __name__ == "__main__":
    # Set seed
    set_seed(42)

    # Get strategy
    strategy = get_strategy()

    print("REPLICAS:", strategy.num_replicas_in_sync)

    # Define global variables
    GLOBAL_BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    IMG_DIM = 224
    CHANNEL_DIM = 25
    EPOCHS = 10
    STEPS_PER_EPOCH = 500
    VALIDATION_STEP = 200

    # Get model and variables
    with strategy.scope():
        model, optimizer, loss_func, transf_points, training_loss, validation_loss = get_model_and_variables(GLOBAL_BATCH_SIZE, IMG_DIM, CHANNEL_DIM)

    # Load datasets (assuming train_files and valid_files are defined)
    train_dataset = get_dataset(train_files, training=True)
    valid_dataset = get_dataset(valid_files, training=False)

    # Train the model
    train_model(train_dataset, valid_dataset, model, optimizer, loss_func, transf_points, training_loss, validation_loss, EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEP)