pip install -q tensorflow-recommenders
pip install -q --upgrade tensorflow-datasets

from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

2022-12-14 12:07:19.595508: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
2022-12-14 12:07:19.595615: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
2022-12-14 12:07:19.595626: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
Read the data

# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])

WARNING:tensorflow:From /tmpfs/src/tf_docs_env/lib/python3.9/site-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
WARNING:tensorflow:From /tmpfs/src/tf_docs_env/lib/python3.9/site-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
Build vocabularies to convert user ids and movie titles into integer indices for embedding layers:


user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)
Define a model
We can define a TFRS model by inheriting from tfrs.Model and implementing the compute_loss method:


class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(user_embeddings, movie_embeddings)
Define the two models and the retrieval task.


# Define user and movie models.
user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
])
movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
  )
)

WARNING:tensorflow:vocab_size is deprecated, please use vocabulary_size.
WARNING:tensorflow:vocab_size is deprecated, please use vocabulary_size.
WARNING:tensorflow:vocab_size is deprecated, please use vocabulary_size.
WARNING:tensorflow:vocab_size is deprecated, please use vocabulary_size.
Fit and evaluate it.
Create the model, train it, and generate predictions:


# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(ratings.batch(4096), epochs=3)

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movies.batch(100).map(lambda title: (title, model.movie_model(title))))

# Get some recommendations.
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")

Epoch 1/3
25/25 [==============================] - 8s 200ms/step - factorized_top_k/top_1_categorical_accuracy: 1.9000e-04 - factorized_top_k/top_5_categorical_accuracy: 0.0024 - factorized_top_k/top_10_categorical_accuracy: 0.0066 - factorized_top_k/top_50_categorical_accuracy: 0.0518 - factorized_top_k/top_100_categorical_accuracy: 0.1124 - loss: 33099.9444 - regularization_loss: 0.0000e+00 - total_loss: 33099.9444
Epoch 2/3
25/25 [==============================] - 5s 192ms/step - factorized_top_k/top_1_categorical_accuracy: 1.9000e-04 - factorized_top_k/top_5_categorical_accuracy: 0.0052 - factorized_top_k/top_10_categorical_accuracy: 0.0143 - factorized_top_k/top_50_categorical_accuracy: 0.1039 - factorized_top_k/top_100_categorical_accuracy: 0.2098 - loss: 31008.8453 - regularization_loss: 0.0000e+00 - total_loss: 31008.8453
Epoch 3/3
25/25 [==============================] - 5s 193ms/step - factorized_top_k/top_1_categorical_accuracy: 3.3000e-04 - factorized_top_k/top_5_categorical_accuracy: 0.0082 - factorized_top_k/top_10_categorical_accuracy: 0.0219 - factorized_top_k/top_50_categorical_accuracy: 0.1439 - factorized_top_k/top_100_categorical_accuracy: 0.2670 - loss: 30420.3803 - regularization_loss: 0.0000e+00 - total_loss: 30420.3803
Top 3 recommendations for user 42: [b'Rent-a-Kid (1995)' b'Just Cause (1995)' b'Aristocats, The (1970)']
