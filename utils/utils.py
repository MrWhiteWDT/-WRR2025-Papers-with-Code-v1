import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tqdm
def fix_random_seed(seed):
    try:
        np.random.seed(seed)
    except NameError:
        print(
            "Warning: Numpy is not imported. Setting the seed for Numpy failed."
        )
    try:
        tf.random.set_seed(seed)
    except NameError:
        print(
            "Warning: TensorFlow is not imported. Setting the seed for TensorFlow failed."
        )
    try:
        random.seed(seed)
    except NameError:
        print(
            "Warning: random module is not imported. Setting the seed for random failed."
        )

def sliding_window(train, sw_width=7, n_out=7, in_start=0):
    data = train.reshape((train.shape[0], 1)) 
    X, y = [], []

    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out

        if out_end <= len(data):

            train_seq = data[in_start:in_end, 0]
            train_seq = train_seq.reshape((len(train_seq), 1))
            X.append(train_seq)
            y.append(data[in_end:out_end, 0])
        in_start += 1

    return np.array(X), np.array(y)

def cal_metrics(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return r2, mse, mae, mape


def rolling_forecast(model, dataset,input_sequence_length = 672,horizon = 168 ,data_name = 'DMA_B(L/s)'):

    input_sequence_length = input_sequence_length 
    slide_step = horizon
    if horizon == 168:
        dataset = dataset.iloc[:,-4704:]
    test_series = dataset[f'{data_name}(L/s)'].values


    all_predictions = []
    all_actuals = []
    all_indices = []

    for i in tqdm.tqdm(range(0, len(test_series) - input_sequence_length - horizon + 1, slide_step)):

        input_start = i
        input_end = i + input_sequence_length

        actual_start = input_end
        actual_end = actual_start + horizon

        current_sequence = list(test_series[input_start:input_end])

        current_window_predictions = []

        for j in range(horizon):
            input_array = np.array(current_sequence).reshape(1, input_sequence_length)

            predicted_value = model.predict(input_array, verbose=0)[0][0]
            current_window_predictions.append(predicted_value)

            current_sequence.pop(0)
            current_sequence.append(predicted_value)

        all_predictions.extend(current_window_predictions)
        all_actuals.extend(test_series[actual_start:actual_end])
        all_indices.extend(dataset.index[actual_start:actual_end])

    results_df = pd.DataFrame({
        'Date': all_indices,
        'Actual': all_actuals,
        'Predicted': all_predictions
    }).set_index('Date')

    return results_df


def rolling_forecast_new(model, dataset, input_sequence_length=672, horizon=168, data_name='DMA_B(L/s)'):

    slide_step = horizon
    test_series = dataset[f'{data_name}(L/s)'].values

    all_predictions = []
    all_actuals = []
    all_indices = []

    last_possible_start = len(test_series) - input_sequence_length - horizon


    for i in tqdm.tqdm(range(last_possible_start, -1, -slide_step)):
        input_start = i
        input_end = i + input_sequence_length
        actual_start = input_end
        actual_end = actual_start + horizon

        current_sequence = list(test_series[input_start:input_end])
        current_window_predictions = []

        # 这是内层循环
        for j in range(horizon):
            input_array = np.array(current_sequence).reshape(1, input_sequence_length)

            predicted_value = model.predict(input_array, verbose=0)[0][0]
            current_window_predictions.append(predicted_value)

            current_sequence.pop(0)
            current_sequence.append(predicted_value)

        all_predictions = current_window_predictions + all_predictions
        all_actuals = list(test_series[actual_start:actual_end]) + all_actuals
        all_indices = list(dataset.index[actual_start:actual_end]) + all_indices

    results_df = pd.DataFrame({
        'Date': all_indices,
        'Actual': all_actuals,
        'Predicted': all_predictions
    }).set_index('Date')

    return results_df