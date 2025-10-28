import keras
from keras import layers, Model
keras.losses.Huber()
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS,TSMixer,Informer,NLinear
from neuralforecast.losses.pytorch import MAE,HuberLoss







def LSTM(input_data=672,output_number=1):
    units_1 = 120
    units_2 = 72
    units_3 = 120
    units_4 = 24
    activation = 'gelu'

    flow_input = layers.Input(shape=(input_data), name="flow_input")
    x = layers.Reshape([1, input_data])(flow_input)
    x = layers.LSTM(units=units_1, return_sequences=True, activation=activation)(x)
    x = layers.LSTM(units=units_2, return_sequences=False, activation=activation)(x)
    x = layers.Dense(units=units_3, activation=activation)(x)
    x = layers.Dense(units=units_4, activation=activation)(x)
    # output data
    output = layers.Dense(output_number, activation=activation, name='Output')(x)
    ##model####
    model = keras.models.Model(inputs=flow_input, outputs=output)
    return model

def CNN(input_data=[672,1],output_number=1):
    input_shape = input_data # [168,1]
    activation = 'relu'

    inputs = keras.Input(shape=(input_shape))  # [None, 10, 12]
    x = layers.Reshape(target_shape=(inputs.shape[1], inputs.shape[2]))(inputs)
    x = layers.Conv1D(filters = 24 , kernel_size= 4, strides= 3, padding='same', use_bias=False,activation=activation)(x)
    x = layers.MaxPool1D(pool_size=4)(x)
    x= layers.Flatten()(x)
    x = layers.Dense(units = 144, activation = activation)(x)
    x = layers.Dense(units = 48, activation = activation)(x)

    outputs = layers.Dense(output_number)(x)

    model_CNN = keras.Model(inputs, outputs)
    return model_CNN

def Conv_LSTM(input_data=[672,1],output_number=1):
    activation = 'gelu'
    input_shape =input_data # [672,1]
    inputs = keras.Input(shape=(input_shape))
    x = layers.Reshape(target_shape =[1,168,4])(inputs)
    x = layers.ConvLSTM1D(filters = 48,kernel_size= 2, strides= 1, padding='same', use_bias=True,activation = activation)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x) 
    x = layers.MaxPool1D(pool_size=4)(x)
    x = layers.LSTM(units=144,return_sequences=True, activation = activation)(x)
    x = layers.Dense(units=24,activation = activation)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_number)(x)
    model = keras.Model(inputs, outputs)
    return model


def NHits_m(input_data=672,output_number=1,max_epoch= 1000):
    
    NHITS_model = NHITS(h=output_number,                           
                    input_size=input_data,       
                    loss=HuberLoss(),            
                    max_steps=max_epoch,         
                    dropout_prob_theta=0,
                    interpolation_mode='linear',
                    stack_types=['identity']*3,
                    n_pool_kernel_size = [2,2,2],
                    n_freq_downsample=[168, 24, 1],
                    val_check_steps=10,            
                    )
    return NHITS_model

def TSMixer_m(input_data=672,output_number=1,max_epoch= 1000):
    
    TSMixer_model = TSMixer(h=output_number,
                      n_series=1,                          
                      input_size=input_data,
                      dropout=0,
                      n_block=5,
                      loss = MAE(),
                      max_steps=max_epoch,                      
                     )

    
    return TSMixer_model

def Informer_m(input_data=672,output_number=1,max_epoch=1000):
    
    Informer_model = Informer(h=output_number,
                       input_size=input_data,
                       hidden_size = 32,
                       conv_hidden_size=32,
                       dropout = 0,
                       n_head=2,
                       encoder_layers = 2,
                       decoder_layers = 1,
                       loss=HuberLoss(),
                       scaler_type='minmax',
                       learning_rate=1e-4,
                       max_steps=max_epoch,
                       val_check_steps=50,
                        distil= False)

    return Informer_model


def NLinear_m(input_data=672,output_number=1,max_epoch=1000):
    
    model = NLinear(
    h=output_number,          
    input_size=input_data,    
    loss=MAE(),               
    scaler_type='robust',     
    learning_rate=1e-3,       
    max_steps=max_epoch,      
    val_check_steps=50,       
    )
    
    return model