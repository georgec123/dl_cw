from sklearn.metrics import accuracy_score
import os
import json
from tensorflow import keras
from keras.callbacks import History 

import pandas as pd
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt
import numpy as np


def get_features(df):
    prev_mid_m_cols = ['c1', 'c2', 'c3', 'c4', 'c5']
    
    # engineer features
    df['ss_lob_12_v'] = df['ss_lob_1_v']+df['ss_lob_2_v']
    df['bs_lob_12_v'] = df['bs_lob_1_v']+df['bs_lob_2_v']

    df['l1_diff']= df['ss_lob_1_v']-df['bs_lob_1_v']
    df['l12_diff']= df['ss_lob_12_v']-df['bs_lob_12_v']

    df['vwap'] = (df['bs_lob_1_p']*df['bs_lob_1_v'] + df['ss_lob_1_p']*df['ss_lob_1_v'])/(df['ss_lob_1_v']+df['bs_lob_1_v'])


    df['bs_pressure1'] = (df['bs_lob_1_v']>df['ss_lob_1_v']).astype(int)

    df['avg_5'] = df[prev_mid_m_cols].mean(axis=1)
    df['momentum_up'] = (df['avg_5']>0.5).astype(int)

    engineered_cols = ['ss_lob_12_v','bs_lob_12_v','l1_diff','l12_diff', 'vwap']

    return df, engineered_cols


def predict_and_accuracy(model,  X_train, X_test, y_train, y_test):
    
    y_train_pred = model.predict(X_train, verbose=0)
    y_train_pred = (y_train_pred>0.5).astype(int)
    train_accuracy = accuracy_score(y_train_pred.flatten(), y_train.to_numpy().flatten())

    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred = (y_test_pred>0.5).astype(int)
    test_accuracy = accuracy_score(y_test_pred.flatten(), y_test.to_numpy().flatten())

    return train_accuracy, test_accuracy

def train_and_test_nn(model, X_train, X_test, y_train, y_test, history):
    """
    Train model and return train and test accuracy
    """
    # reset model weights

    y_train_pred = model.predict(X_train)
    y_train_pred = (y_train_pred>0.5).astype(int)
    train_accuracy = accuracy_score(y_train_pred.flatten(), y_train.to_numpy().flatten())

    y_test_pred = model.predict(X_test)
    y_test_pred = (y_test_pred>0.5).astype(int)
    test_accuracy = accuracy_score(y_test_pred.flatten(), y_test.to_numpy().flatten())

    return train_accuracy, test_accuracy, history


def log_results(columns: list, eta: float, batch_size: int, epochs: int, optimizer, 
        nodes: int, accuracy, all_history_loss, all_history_accuracy, loss='binary_crossentropy', target='midprice_up', fp='../results.json'):


    results = {
        'columns': columns,
        'eta': eta,
        'batch_size': batch_size,
        'epochs': epochs,
        'optimizer': optimizer.__name__,
        'nodes': nodes,
        'accuracy': accuracy,
        'all_history_loss': all_history_loss,
        'all_history_accuracy': all_history_accuracy,
        'loss': loss
    }

    if not os.path.exists(fp):
        data = []
    else:
        with open(fp, 'r') as f:
            data = json.load(f)

    data+=[results]

    with open(fp, 'w') as f:
        json.dump(data, f, indent=True)
        

def train_predict_log(df: pd.DataFrame, columns: list, eta: float, batch_size: int, 
                epochs: int, optimizer,  nodes: int, fp: str, loss='binary_crossentropy', target='midprice_up'):
        
        X =df[columns] 
        Y = df[target]
        
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(len(X.columns),)),
            keras.layers.Dense(nodes, activation="elu"),
            keras.layers.Dense(1, activation="sigmoid")
            ])

        

        all_accuracy = []
        all_history_loss = []
        all_history_accuracy = []

        kf = KFold(n_splits=5, random_state=2, shuffle=True)
        for train_index, test_index in kf.split(X):
            history = History()

            X_train, X_test = X.loc[train_index,:], X.loc[test_index, :]
            y_train, y_test = Y.loc[train_index], Y.loc[test_index]

            model = keras.models.clone_model(model)
            model.compile(optimizer=optimizer(learning_rate=eta), loss=loss, metrics=["accuracy"])

            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[history])  

            
            train_accuracy, test_accuracy, history = train_and_test_nn(model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                history=history)
            all_accuracy.append((train_accuracy, test_accuracy))
            all_history_loss.append(history.history['loss'])
            all_history_accuracy.append(history.history['accuracy'])


        

        log_results(columns=columns, eta=eta, batch_size=batch_size, epochs=epochs, optimizer=optimizer, 
            nodes=nodes, accuracy=all_accuracy, all_history_loss=all_history_loss, all_history_accuracy=all_history_accuracy, loss=loss,
            target=target, fp=fp)
        


def k_fold(model, X, Y, batch_size, epochs, history, splits=5):
    s = time.time()

    X = X.reset_index(drop=True).copy()
    Y = Y.reset_index(drop=True).copy()

    kf = KFold(n_splits=splits, random_state=2, shuffle=True)

    accuracy = []

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = Y.loc[train_index], Y.loc[test_index]

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[history], verbose=0)

        accuracy.append(predict_and_accuracy(model, X_train=X_train, X_test=X_test, y_train=y_train,
                                y_test=y_test))

    accuracy = np.array(accuracy)

    test_accuracy = accuracy[:,1]
    train_accuracy = accuracy[:,0]

    ax = plt.boxplot([train_accuracy, test_accuracy], labels=['train','test'])
    plt.scatter([2]*len(test_accuracy), test_accuracy);
    plt.scatter([1]*len(train_accuracy), train_accuracy); plt.show()

    e = time.time()

    print(f"Time taken: {e-s:.1f}s")