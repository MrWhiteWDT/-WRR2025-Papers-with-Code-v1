import keras
from keras import layers, Model
from keras.models import load_model
import pandas as pd
import argparse
import sys
from utils.utils import *
from model.LSTM_Mixer import LSTM_Mixer
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import datetime
import tensorflow as tf
import os
from model.other_models import LSTM,CNN,Conv_LSTM,NHits_m,TSMixer_m,Informer_m
from neuralforecast import NeuralForecast
import warnings
def get_arguments(raw_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        default='Conv_LSTM', type=str,
                        choices=['LSTM_Mixer', 'LSTM_Mixer_with_Exog', 'LSTM_Mixer_add','NHITS', 'TSMixer', 'CNN', 'Conv_LSTM', 'Informer', 'LSTM', 'NHits','TSMixer','GRU_Mixer'],
                        help="support model selection only.")


    parser.add_argument('--model_loc',
                        default='F:/Water_demandforecasting_20250829/save_result/Summary/model_save',
                        type=str,
                        help='模型保存地址')

    parser.add_argument('--step_size',
                        default=168,
                        type=int,
                        help='')

    args = parser.parse_args(args=raw_args)
    return args

if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    fix_random_seed(37)
    current_time = datetime.datetime.now()
    # 格式化时间
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 创建文件夹
    os.makedirs(f'.\save_result\Summary/{args.step_size}h/{args.model}/metrics', exist_ok=True)
    os.makedirs(f".\save_result\Summary/{args.step_size}h/{args.model}/results", exist_ok=True)

    for DMA_name in ['DMA_I']:

        print(f'{args.model_loc}/{args.model}/{DMA_name}.h5')

        model = load_model(f'{args.model_loc}/{args.model}/{DMA_name}.h5')

        dataset = pd.read_excel(f'F:/Water_demandforecasting_20250829/input/testdataset/{DMA_name}.xlsx')

        results_df = rolling_forecast_new(model=model,
                                          input_sequence_length = 672,
                                          horizon = args.step_size,
                                          dataset = dataset,
                                          data_name = DMA_name)

        r2 = r2_score(results_df['Actual'], results_df['Predicted'])
        mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
        mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
        mape = mean_absolute_percentage_error(results_df['Actual'], results_df['Predicted'])
        results_df.to_excel(f".\save_result\Summary/{args.step_size}h/{args.model}/results/{DMA_name}.xlsx")
        metrics = pd.DataFrame(index=['r2', 'mse', 'mae', 'mape'], columns=[f'{DMA_name}'])
        metrics.loc['r2'] = r2
        metrics.loc['mse'] = mse
        metrics.loc['mae'] = mae
        metrics.loc['mape'] = mape

        metrics.round(2).to_excel(f".\save_result\Summary/{args.step_size}h/{args.model}/metrics/{DMA_name}.xlsx")





