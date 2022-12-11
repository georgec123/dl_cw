import pandas as pd
import numpy as np

def organise_data(split):
    df = pd.read_csv('./data/Data_A.csv', header=None)
    df.columns = ['label', 's_1_p', 's_1_v', 'b_1_p', 'b_1_v', 's_2_p', 's_2_v', 'b_2_p', 'b_2_v', 's_3_p', 's_3_v',
                  'b_3_p', 'b_3_v', 's_4_p', 's_4_v', 'b_4_p', 'b_4_v', 'mid_1', 'mid_2', 'mid_3', 'mid_4', 'mid_5']
    train = df[:split]
    test = df[split:].reset_index()
    test = test.drop(['index'], axis=1)

    numeric_names = ['s_1_p', 's_1_v', 'b_1_p', 'b_1_v', 's_2_p', 's_2_v', 'b_2_p', 'b_2_v', 's_3_p', 's_3_v',
                  'b_3_p', 'b_3_v', 's_4_p', 's_4_v', 'b_4_p', 'b_4_v']
    binary_names = ['mid_1', 'mid_2', 'mid_3', 'mid_4', 'mid_5']
    
    return [train, test, numeric_names, binary_names]

#arr = organise_data(90000)
#print(arr[1].iloc[:, 1:23].to_numpy())
#print(arr[0]['label'])
#print(arr[1])

def accuracy(list1, list2):
    return np.mean((list1.flatten()==list2))