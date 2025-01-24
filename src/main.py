from train_tpu import train_model
from utils import get_dataset
from config import GLOBAL_BATCH_SIZE, IMG_DIM, CHANNEL_DIM, EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEP

# Load datasets (assuming train_files and valid_files are defined)
train_dataset = get_dataset(train_files, training=True)
valid_dataset = get_dataset(valid_files, training=False)

# Train the model
train_model(train_dataset, valid_dataset, model, optimizer, loss_func, transf_points, training_loss, validation_loss, EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEP)