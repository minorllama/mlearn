"""

Compiling and training the same neural network in TensorFlow and PyTorch.

In graph mode, TensorFlow traces the training step with tf.function and PyTorch compiles the model with TorchInductor. XLA mode adds TensorFlow
XLA compilation. PyTorch's Linux CPU compiler needs a C++ compiler such as g++ on PATH. The first compiled batch can take substantially longer.

Example:

    env -u TF_USE_LEGACY_KERAS -u PYTHONPATH -u PYTHONHOME venv-tfq312/bin/python model_compilation_tensorflow_torch.py eager

Mode: eager
      |          TensorFlow          |           PyTorch
Epoch | train loss  test loss    acc | train loss  test loss    acc
#     1 |     0.6600     0.6318  66.2% | #     0.6600     0.6318  66.2%
#     2 |     0.6134     0.5900  83.6% | #     0.6134     0.5900  83.6%
#     3 |     0.5723     0.5501  94.7% | #     0.5723     0.5501  94.7%
#     4 |     0.5330     0.5110  97.1% | #     0.5330     0.5110  97.1%
#     5 |     0.4938     0.4721  98.6% | #     0.4938     0.4721  98.6%
#     6 |     0.4547     0.4334  99.0% | #     0.4547     0.4334  99.0%
#     7 |     0.4167     0.3951  99.0% | #     0.4167     0.3951  99.0%
#     8 |     0.3797     0.3595  98.8% | #     0.3797     0.3595  98.8%
#     9 |     0.3454     0.3261  99.4% | #     0.3454     0.3261  99.4%
#    10 |     0.3133     0.2955  99.4% | #     0.3133     0.2955  99.4%
#    11 |     0.2841     0.2683 100.0% | #     0.2841     0.2683 100.0%
#    12 |     0.2577     0.2431 100.0% | #     0.2577     0.2431 100.0%
#    13 |     0.2338     0.2207 100.0% | #     0.2338     0.2207 100.0%
#    14 |     0.2127     0.2016 100.0% | #     0.2127     0.2016 100.0%
#    15 |     0.1941     0.1840 100.0% | #     0.1941     0.1840 100.0%
#    16 |     0.1776     0.1689 100.0% | #     0.1776     0.1689 100.0%
#    17 |     0.1629     0.1555 100.0% | #     0.1629     0.1555 100.0%
#    18 |     0.1501     0.1427 100.0% | #     0.1501     0.1427 100.0%
#    19 |     0.1386     0.1322 100.0% | #     0.1386     0.1322 100.0%
#    20 |     0.1284     0.1226 100.0% | #     0.1284     0.1226 100.0%


"""

import sys

import numpy as np
import tensorflow as tf
import torch
from torch import nn


SEED = 7
HIDDEN = 16
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.1


# data shared by both frameworks
def make_data(rng):
    """Return (features, labels) pairs for training and testing."""
    labels = np.repeat(np.arange(2, dtype=np.int64), 1024)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=len(labels))
    radius = 0.55 + 0.65 * labels + rng.normal(0.0, 0.10, size=len(labels))
    features = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    features = features.astype(np.float32)  # shape (2048, 2).
    order = rng.permutation(len(labels))
    train, test = order[:1536], order[1536:]
    return (features[train], labels[train]), (features[test], labels[test])


def as_tf(data):
    x, y = data
    return tf.convert_to_tensor(x), tf.convert_to_tensor(y)


def as_torch(data):
    x, y = data
    # numPy supplies float32 features and int64 class indices (torch.long).
    return torch.from_numpy(x), torch.from_numpy(y)


# model and initial parameters
# 
# both models compute Linear -> ReLU -> Linear. A batch of B coordinate
# pairs has shape (B, 2). The output also has shape (B, 2), with two raw
# class scores (logits) per point. Each label is a class index, 0 or 1.


def make_tf_model():
    # input builds the Dense layers so their weights are available immediately.
    return tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(HIDDEN),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(2),
    ])


def make_torch_model(tf_model):
    model = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2))
    # copy the Dense weights so both runs start at the same point.
    # keras stores (in_features, out_features), PyTorch stores the transpose.
    with torch.no_grad():
        for index in (0, 2):
            kernel, bias = tf_model.layers[index].get_weights()
            model[index].weight.copy_(torch.from_numpy(kernel.T.copy()))
            model[index].bias.copy_(torch.from_numpy(bias))
    return model


# one training step
# 
# both cross-entropy losses take logits and average over the batch.
# the loss handles the conversion from logits to log-probabilities.


def train_tf_step(model, loss_fn, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)  # keras takes labels first.
    params = model.trainable_variables
    gradients = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(gradients, params))
    return loss  # return a tensor so this function can run inside a graph.


def train_torch_step(model, loss_fn, optimizer, x, y):
    optimizer.zero_grad()  # backward() accumulates into each parameter's .grad.
    logits = model(x)
    loss = loss_fn(logits, y)  # pyTorch takes logits first.
    loss.backward()
    optimizer.step()  # reads the gradients stored in the parameters.
    return loss.item()


# evaluation on the held-out test set
# 
# the class with the largest logit is also the class with the largest
# softmax probability. Accuracy is the fraction of correct predictions.


def eval_tf(model, loss_fn, x, y):
    # the forward pass runs outside a GradientTape.
    logits = model(x, training=False)
    loss = loss_fn(y, logits)
    predicted = tf.argmax(logits, axis=1, output_type=y.dtype)
    accuracy = tf.reduce_mean(tf.cast(predicted == y, tf.float32))
    return float(loss.numpy()), float(accuracy.numpy())


@torch.no_grad()
def eval_torch(model, loss_fn, x, y):
    model.eval()  # sets inference behavior for layers such as dropout.
    logits = model(x)
    loss = loss_fn(logits, y)
    accuracy = (logits.argmax(dim=1) == y).float().mean()
    return loss.item(), accuracy.item()


# select eager or compiled execution.


def make_tf_step(model, loss_fn, optimizer, mode):
    def step(x, y):
        return train_tf_step(model, loss_fn, optimizer, x, y)

    if mode == "eager":
        return step

    # create optimizer variables before tracing. Capture the forward pass,
    # cross-entropy, gradient calculation, and SGD update in one graph.
    optimizer.build(model.trainable_variables)
    return tf.function(step, jit_compile=(mode == "xla"))


def make_torch_train_model(model, mode):
    if mode == "eager":
        return model

    # inductor compiles the model forward and its corresponding backward.
    # the loss function and Python optimizer calls stay outside this boundary.
    return torch.compile(model, backend="inductor")


# train in lockstep and report the two runs side by side.
def train_epoch(tf_step, torch_setup, tf_train, torch_train, order):
    tf_x, tf_y = tf_train
    torch_x, torch_y = torch_train
    tf_total = torch_total = 0.0

    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start : start + BATCH_SIZE]
        tf_loss = tf_step(tf.gather(tf_x, batch), tf.gather(tf_y, batch))
        torch_loss = train_torch_step(*torch_setup, torch_x[batch], torch_y[batch])
        # weight each batch by its size to get the mean over all examples.
        tf_total += float(tf_loss.numpy()) * len(batch)
        torch_total += torch_loss * len(batch)

    return tf_total / len(order), torch_total / len(order)


def run(mode):
    tf.keras.utils.set_random_seed(SEED)
    torch.manual_seed(SEED)
    # small batches run quickly with one PyTorch CPU thread.
    torch.set_num_threads(1)
    data_rng = np.random.default_rng(SEED)
    batch_rng = np.random.default_rng(SEED + 1)
    train, test = make_data(data_rng)
    tf_train, torch_train = as_tf(train), as_torch(train)
    tf_test, torch_test = as_tf(test), as_torch(test)

    tf_model = make_tf_model()
    torch_model = make_torch_model(tf_model)
    tf_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    torch_loss_fn = nn.CrossEntropyLoss()
    tf_optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=LEARNING_RATE)
    tf_step = make_tf_step(tf_model, tf_loss_fn, tf_optimizer, mode)
    torch_train_model = make_torch_train_model(torch_model, mode)
    torch_setup = (torch_train_model, torch_loss_fn, torch_optimizer)

    print(f"Mode: {mode}")
    print("      |          TensorFlow          |           PyTorch")
    print("Epoch | train loss  test loss    acc | train loss  test loss    acc")
    for epoch in range(1, EPOCHS + 1):
        torch_model.train()  # restore training mode after the previous evaluation.
        order = batch_rng.permutation(len(train[1]))
        tf_train_loss, torch_train_loss = train_epoch(
            tf_step, torch_setup, tf_train, torch_train, order
        )
        # evaluate both original models eagerly with the updated weights.
        tf_test_loss, tf_acc = eval_tf(tf_model, tf_loss_fn, *tf_test)
        torch_test_loss, torch_acc = eval_torch(torch_model, torch_loss_fn, *torch_test)
        print(
            f"# {epoch:5d} | {tf_train_loss:10.4f} {tf_test_loss:10.4f} {tf_acc:6.1%} | "
            f"# {torch_train_loss:10.4f} {torch_test_loss:10.4f} {torch_acc:6.1%}"
        )


if __name__ == "__main__":
    try:
        mode = sys.argv[1]
        if not mode in ["eager", "graph", "xla"]:
            raise Exception("malformed arguments")
    except:
        usage = f"{sys.argv[0]} eager|graph|xla"
        print(usage)
        sys.exit(1)

    with tf.device("/CPU:0"):
        run(mode)



