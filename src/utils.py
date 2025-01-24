import random
import numpy as np
import torch
import os
import tensorflow as tf

def neg_multi_log_likelihood(gt, pred, confidences, avails):
    assert len(pred.shape) == 4, f"expected 4D (B,M,T,C) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape
    
    assert gt.shape == (batch_size, future_len, num_coords), f"expected 3D (Batch, Time, Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 2D (Batch, Modes) array for gt, got {confidences.shape}"
    assert avails.shape == (batch_size, future_len), f"expected 2D (Batch, Time) array for gt, got {avails.shape}"
    
    gt = tf.expand_dims(gt, axis=1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords
    error = tf.math.reduce_sum(((gt - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    confidences = tf.clip_by_value(confidences, clip_value_min=tf.pow(0.1, 12), clip_value_max=1.0)  # to avoid exploding gradient
    error = tf.math.log(confidences) - 0.5 * tf.math.reduce_sum(error, axis=-1)  # reduce time
    
    max_value = tf.math.reduce_max(error, axis=1, keepdims=True)  # error are negative at this point, so max() gives the minimum one
    error = -tf.math.log(tf.math.reduce_sum(tf.exp(error - max_value), axis=-1, keepdims=True)) - max_value  # reduce modes
    return error

def transform_points(points, transf_matrix):
    transf_matrix = tf.expand_dims(transf_matrix, axis=-1)
    assert len(points.shape) == len(transf_matrix.shape) == 4, (
        f"dimensions mismatch, both points ({points.shape}) and "
        f"transf_matrix ({transf_matrix.shape}) needs to be tensors of rank 4."
    )
    
    if points.shape[3] not in [2, 3]:
        raise AssertionError(f"Points input should be (N, 2) or (N, 3) shape, received {points.shape}")
    
    assert points.shape[3] == transf_matrix.shape[2] - 1, "points dim should be one less than matrix dim"
    
    points = tf.cast(points, tf.float64)
    points = tf.matmul(points, tf.transpose(transf_matrix[:, :-1, :-1, :], perm=[0, 3, 2, 1]))
    return tf.cast(points, tf.float32)

def load_dataset(filenames, labeled=True, ordered=False, training=True):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    dataset = dataset.map(read_train_tfrecord if training else read_validation_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_dataset(files, training=True):
    if training:
        dataset = load_dataset(files, training=True)
        dataset = dataset.shuffle(128).prefetch(AUTO)
    else:
        dataset = load_dataset(files, training=False)
        dataset = dataset.prefetch(AUTO)
    return dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(42)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def get_strategy():
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)