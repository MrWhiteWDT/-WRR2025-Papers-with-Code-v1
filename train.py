import keras
from keras import layers, Model
import pandas as pd
import argparse
import sys
from utils.utils import *
from model.LSTM_Mixer import LSTM_Mixer
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import datetime
import tensorflow as tf
import os
from model.other_models import LSTM,CNN,Conv_LSTM,NHits_m,TSMixer_m,Informer_m,NLinear_m
from neuralforecast import NeuralForecast
import warnings

# 忽略可能的警告
warnings.filterwarnings('ignore')

def get_arguments(raw_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        default='CNN', type=str,
                        choices=['LSTM_Mixer', 'LSTM_Mixer_with_Exog', 'LSTM_Mixer_add','NHITS', 'TSMixer', 'CNN', 'Conv_LSTM', 'Informer', 'LSTM', 'NHits','TSMixer','GRU_Mixer', 'NLinear'],
                        help="support model selection only.")

    parser.add_argument('--DMA_name_list',
                        default=['DMA_D'],
                        type=list,
                        help="DMA name.")

    parser.add_argument('--Exog_consider',
                        default=False,
                        type=bool,
                        help="Whether exogenous variables are used")

    parser.add_argument('--epochs',
                        default=50, type=int,
                        help="epochs.")

    parser.add_argument('--batch_size',
                        default=32, type=int,
                        help="epochs.")

    parser.add_argument('--lr',
                        default=5e-04, type=float,
                        help="seed")

    parser.add_argument('--activate_function',
                        default='gelu', type=str,
                        help='gelu,relu,tanh')

    parser.add_argument('--early_stop',
                        default=True, type=bool,
                        help="Whether to use an early stop strategy")

    parser.add_argument('--step_size',
                        default=1,
                        type=int,
                        help='')

    parser.add_argument('--input_lenth',
                        default= 168*4,
                        type= int,
    )

    args = parser.parse_args(args=raw_args)
    return args

def read_excel_safe(file_path):

    try:
        return pd.read_excel(file_path)
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError encountered: {e}")
        print(f"Trying with different encoding for file: {file_path}")
        try:
            return pd.read_excel(file_path, engine='openpyxl')
        except Exception as e2:
            print(f"Error with openpyxl engine: {e2}")
            try:
                return pd.read_excel(file_path, engine='xlrd')
            except Exception as e3:
                print(f"Error with xlrd engine: {e3}")



if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    fix_random_seed(1399)
    current_time = datetime.datetime.now()
    
    # 格式化时间
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 创建文件夹
    os.makedirs(f"./save_result/{args.model}/{formatted_time}", exist_ok=True)
    os.makedirs(f"./save_result/{args.model}/{formatted_time}/metrics", exist_ok=True)
    os.makedirs(f"./save_result/{args.model}/{formatted_time}/results", exist_ok=True)
    os.makedirs(f"./model_save/{args.model}/{formatted_time}", exist_ok=True)

    for DMA_name in args.DMA_name_list:

        # 读取数据
        try:
            train_data = read_excel_safe(f'./input/traindataset/{DMA_name}.xlsx')
            test_data = read_excel_safe(f'./input/testdataset/{DMA_name}.xlsx')
        except Exception as e:
            print(f"读取 {DMA_name} 数据时出错: {e}")
            continue
            
        total_data = pd.concat([train_data, test_data], axis=0)

        total_data = total_data.rename(columns={'Date': 'ds', f'{DMA_name}(L/s)': 'y'})
        total_data['unique_id'] = 1
        total_data = total_data[['unique_id', 'ds', 'y']]


        # 读取模型
        if args.model == 'LSTM_Mixer':
            model = LSTM_Mixer(input_shape=(args.input_lenth,), output_number=args.step_size)
        elif args.model == 'LSTM':
            model = LSTM(input_data=args.input_lenth, output_number=args.step_size)
        elif args.model == 'CNN':
            model = CNN(input_data=[args.input_lenth,1], output_number=args.step_size)
        elif args.model == 'Conv_LSTM':
            model = Conv_LSTM(input_data=[args.input_lenth,1], output_number=args.step_size)
        elif args.model == 'NHITS':
            model = NHits_m(input_data = args.input_lenth, output_number=args.step_size, max_epoch=args.epochs)
        elif args.model == 'TSMixer':
            model = TSMixer_m(input_data = args.input_lenth, output_number=args.step_size, max_epoch=args.epochs)
        elif args.model == 'Informer':
            model = Informer_m(input_data = args.input_lenth, output_number=args.step_size, max_epoch=args.epochs)
        elif args.model == 'NLinear':
            model = NLinear_m(input_data=args.input_lenth, output_number=args.step_size, max_epoch=args.epochs)

        if args.model in ['LSTM_Mixer', 'LSTM_Mixer_add','LSTM_Mixer_with_Exog','LSTM','CNN','Conv_LSTM','GRU_Mixer']:
            train_data = np.array(train_data.iloc[:, 1])
            test_data = np.array(test_data.iloc[:, 1])
            train_dataset = sliding_window(train_data, args.input_lenth, args.step_size)
            train_feature = tf.constant(train_dataset[0])
            train_label = tf.constant(train_dataset[1])
            test_dataset = sliding_window(test_data, args.input_lenth, args.step_size)
            test_feature = tf.constant(test_dataset[0])
            test_label = tf.constant(test_dataset[1])

            model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(learning_rate=args.lr),metrics=['mape'])

            batch_size = args.batch_size
            steps_per_epoch = len(train_feature) // 32

            if args.early_stop:
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.0005)
                reduce_lr = keras.callbacks.LearningRateScheduler(lambda x: args.lr * 0.95 ** x)
                model.fit(train_feature,train_label, epochs=args.epochs, steps_per_epoch=steps_per_epoch,callbacks=[stop_early, reduce_lr])
            else:
                model.fit(train_feature, train_label, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
            model.save(f"./model_save/{args.model}/{formatted_time}/{DMA_name}.h5")

            current_time = datetime.datetime.now()
            print("当前时间:", current_time)

            if args.model == 'LSTM_Mixer' or args.model == 'LSTM_Mixer_add' or args.model == 'LSTM_Mixer_with_Exog':
                predict = model.predict(test_feature)
                middle = keras.models.Model(inputs=model.input, outputs=model.get_layer('branch_1_prediction_output').output)
                middle_2 = keras.models.Model(inputs=model.input, outputs=model.get_layer('branch_2_prediction_output').output)
                middle_3 = keras.models.Model(inputs=model.input, outputs=model.get_layer('branch_3_prediction_output').output)

                middle_result_1 = middle.predict(test_feature)
                middle_result_2 = middle_2.predict(test_feature)
                middle_result_3 = middle_3.predict(test_feature)

                save_data1 = pd.DataFrame()
                save_data2 = pd.DataFrame()
                save_data3 = pd.DataFrame()
                save_data4 = pd.DataFrame()

                if args.step_size == 1:
                    save_data1['predictions'] = predict.reshape(predict.shape[0])
                    save_data2['predictions'] = middle_result_1.reshape(predict.shape[0])
                    save_data3['predictions'] = middle_result_2.reshape(predict.shape[0])
                    save_data4['predictions'] = middle_result_3.reshape(predict.shape[0])
                    
                    save_data1.to_excel(
                        f"./save_result/{args.model}/{formatted_time}/results/{DMA_name}_{args.step_size}_predictions.xlsx",
                        index=False)
                    save_data2.to_excel(
                        f"./save_result/{args.model}/{formatted_time}/results/{DMA_name}_{args.step_size}_Week_output.xlsx",
                        index=False)
                    save_data3.to_excel(
                        f"./save_result/{args.model}/{formatted_time}/results/{DMA_name}_{args.step_size}_Day_output.xlsx",
                        index=False)
                    save_data4.to_excel(
                        f"./save_result/{args.model}/{formatted_time}/results/{DMA_name}_{args.step_size}_Hour_output.xlsx",
                        index=False)

                else:
                    for i in range(1, args.step_size+1):
                        save_data1[f'{i}_step'] = predict[:, i - 1]
                        save_data1.to_excel(f"./save_result/{args.model}/{formatted_time}/results/{DMA_name}_{args.step_size}_predictions.xlsx", index=False)

            # 计算评价指标
                r2 = r2_score(test_label, predict)
                mse = mean_squared_error(test_label, predict)
                mae = mean_absolute_error(test_label, predict)
                mape = mean_absolute_percentage_error(test_label, predict)
                metrics = pd.DataFrame(index=['r2', 'mse', 'mae', 'mape'], columns=[f'{DMA_name}'])
                metrics.loc['r2'] = r2
                metrics.loc['mse'] = mse
                metrics.loc['mae'] = mae
                metrics.loc['mape'] = mape
                print(metrics)
                metrics.round(2).to_excel(f"./save_result/{args.model}/{formatted_time}/metrics/{DMA_name}_{args.step_size}.xlsx")

            else:
                predict = model.predict(test_feature)
                save_data1 = pd.DataFrame()

                if args.step_size == 1:
                    save_data1['predictions'] = predict.reshape(predict.shape[0])

                else:
                    predict = predict.reshape(predict.shape[0], predict.shape[1])
                    for i in range(1, args.step_size + 1):
                        save_data1[f'{i}_step'] = predict[:, i - 1]

                save_data1.to_excel(f"./save_result/{args.model}/{formatted_time}/results/{DMA_name}.xlsx",index=False)
                r2 = r2_score(test_label, predict)
                mse = mean_squared_error(test_label, predict)
                mae = mean_absolute_error(test_label, predict)
                mape = mean_absolute_percentage_error(test_label, predict)
                metrics = pd.DataFrame(index=['r2', 'mse', 'mae', 'mape'], columns=[f'{DMA_name}'])
                metrics.loc['r2'] = r2
                metrics.loc['mse'] = mse
                metrics.loc['mae'] = mae
                metrics.loc['mape'] = mape
                metrics.round(2).to_excel(f"./save_result/{args.model}/{formatted_time}/metrics/{DMA_name}.xlsx")


        if args.model in ['NHITS', 'TSMixer', 'Informer','NLinear']:

            mf = NeuralForecast(models=[model], freq='H')

            predict = mf.cross_validation(df = total_data, val_size=0, test_size=len(test_data)//args.step_size*args.step_size,n_windows=None, step_size=args.step_size,
                                         verbose=0, refit=False,)
            predict.to_excel(f"./save_result/{args.model}/{formatted_time}/results/{DMA_name}.xlsx")

            pre_result = np.array(predict.iloc[:,3]).flatten()
            real_result = np.array(predict.iloc[:,4]).flatten()

            r2 = r2_score(pre_result ,real_result)
            mse = mean_squared_error(pre_result ,real_result)
            mae = mean_absolute_error(pre_result ,real_result)
            mape = mean_absolute_percentage_error(pre_result ,real_result)

            metrics = pd.DataFrame(index=['r2', 'mse', 'mae', 'mape'], columns=[f'{DMA_name}'])
            metrics.loc['r2'] = r2
            metrics.loc['mse'] = mse
            metrics.loc['mae'] = mae
            metrics.loc['mape'] = mape
            metrics.round(2).to_excel(f"./save_result/{args.model}/{formatted_time}/metrics/{DMA_name}.xlsx")