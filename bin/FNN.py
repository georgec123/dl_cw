import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import pandas as pf
import matplotlib.pyplot as plt
from Data import organise_data, accuracy, B_data

arr = organise_data(80000)
train = arr[0]
test = arr[1]
numeric_names = arr[2]
binary_names = arr[3]

n = 100

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)

def preprocess(df, numeric_names, binary_names):
    target = df.pop('label')
    numeric_features = df[numeric_names]
    binary_features = df[binary_names]
    tf.convert_to_tensor(numeric_features)

    inputs = {}
    for name, column in df.items():
        if type(column[0]) == str:
            dtype = tf.string
        elif (name in binary_names):
            dtype = tf.int64
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

    preprocessed = []

    for name in binary_names:
        inp = inputs[name]
        inp = inp[:, tf.newaxis]
        float_value = tf.cast(inp, tf.float32)
        preprocessed.append(float_value)
    
    preprocessed

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(numeric_features))) 
    numeric_inputs = {}
    for name in numeric_names:
        numeric_inputs[name]=inputs[name]

    numeric_inputs = stack_dict(numeric_inputs)
    numeric_normalized = normalizer(numeric_inputs)

    preprocessed.append(numeric_normalized)

    preprocesssed_result = tf.concat(preprocessed, axis=-1)
    preprocesssed_result
    preprocessor = tf.keras.Model(inputs, preprocesssed_result)
    x = preprocessor(inputs)

    return [x, inputs, target]

def fit_FNN(df, numeric_names, binary_names):
    pre = preprocess(df, numeric_names, binary_names)
    x = pre[0]
    inputs = pre[1]
    target = pre[2]

    f = keras.Sequential([
        keras.layers.InputLayer(input_shape=(21,)),
        keras.layers.Dense(n, activation="elu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    res = f(x)

    model = tf.keras.Model(inputs, res)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    callbacks_used = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy',factor=0.2,patience=5,min_lr=0)
    ]
    
    model.fit(dict(df), target, epochs=5, batch_size=100, callbacks=callbacks_used)

    return model

def predict(f, df):

    Y_hat = (f.predict(dict(df)) > 0.5).astype("int32")

    return Y_hat

h = fit_FNN(train, numeric_names, binary_names)
label = test['label'].tolist()
test = test.drop(['label'], axis=1)
Y_hat = predict(h, test, numeric_names, binary_names)

B_data = B_data()
print(test)
print(B_data)
Y_hat_B = predict(h, B_data, numeric_names, binary_names)
print(Y_hat)
print(accuracy(Y_hat, label))
print(Y_hat_B)
