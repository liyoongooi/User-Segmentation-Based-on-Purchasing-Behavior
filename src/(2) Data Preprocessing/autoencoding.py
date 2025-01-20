import numpy as np
import pandas as pd
import tensorflow as tf
import random

from keras.layers import Input, Dense
from keras.models import Model

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
random_state = seed

def autoencoding(data, act='relu', batch_size=64, epochs=10):
    X = data.values
    dims=[X.shape[-1], 500, 500, 2000, 10]
    
    input_layer = Input(shape=(dims[0],), name='input')
    h = input_layer

    for i in range(len(dims) - 2):
        h = Dense(dims[i + 1], activation=act, name=f'encoder_{i}')(h)

    h = Dense(dims[-1], name=f'encoder_{len(dims) - 2}')(h)

    for i in range(len(dims) - 2, 0, -1):
        h = Dense(dims[i], activation=act, name=f'decoder_{i}')(h)

    h = Dense(dims[0], name='decoder_0')(h)

    autoencoder = Model(inputs=input_layer, outputs=h)
    autoencoder.compile(loss='mse', optimizer='adam')
    
    autoencoder.fit(
        X,
        X,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    encoded_layer = f'encoder_{(len(dims) - 2)}'
    hidden_encoder_layer = autoencoder.get_layer(name=encoded_layer).output
    encoder = Model(inputs=autoencoder.input, outputs=hidden_encoder_layer)
    
    X_encoded = encoder.predict(X)
    encoded_df = pd.DataFrame(X_encoded, columns=[f'encoded_{i}' for i in range(X_encoded.shape[1])], index=data.index)
    
    return encoded_df

