from keras import layers, Model

def LSTM_Mixer(input_shape=(672,), output_number=1):
    # 为整个模型定义单一的输入层
    flow_input = layers.Input(shape=input_shape, name="flow_input")
    
    # 使用 168 的序列长度处理输入数据
    branch_1_output, reconstruction_1 = create_processing_branch(
    input_tensor=flow_input,
    reshape_dims=(168, 4),
    lstm_units=(4, 2, 2, 4),
    dense_units=(42, 21),
    output_number=output_number,
    name_prefix="branch_1" )
   
    # 处理残差 (原始输入 - 分支1的重构输出)，序列长度为 24
    residual_input_2 = layers.subtract([flow_input, reconstruction_1], name="residual_input_2")
    branch_2_output, reconstruction_2 = create_processing_branch(
    input_tensor=residual_input_2,
    reshape_dims=(24, 28),
    lstm_units=(28, 14, 14, 28),
    dense_units=(84, 42),
    output_number=output_number,
    name_prefix="branch_2" )

    # 处理上一层计算出的最终残差，序列长度为 1
    residual_input_3 = layers.subtract([residual_input_2, reconstruction_2], name="residual_input_3")
    branch_3_output = create_final_branch(
    input_tensor=residual_input_3,
    output_number=output_number,
    name_prefix="branch_3" )

    # --- 最终聚合 ---
    # 将三个分支的输出相加，得到最终结果
    final_output = layers.add([branch_1_output, branch_2_output, branch_3_output], name="final_output")
    # 创建并返回 Keras 模型
    model = Model(inputs=flow_input, outputs=final_output, name="multi_branch_lstm_model")
    return model

def create_processing_branch(input_tensor, reshape_dims, lstm_units, dense_units, output_number, name_prefix):
    # 为LSTM层重塑输入张量的形状
    reshaped_input = layers.Reshape(reshape_dims, name=f"{name_prefix}_reshape")(input_tensor)
    # 类似LSTM自编码器的网络栈
    encoder_lstm_1 = layers.LSTM(lstm_units[0], return_sequences=True, activation='gelu',
    name=f"{name_prefix}_encoder_lstm_1")(reshaped_input)
    
    bottleneck_lstm = layers.LSTM(lstm_units[1], return_sequences=True, activation='gelu',
    name=f"{name_prefix}_bottleneck_lstm")(encoder_lstm_1)
    
    decoder_lstm_1 = layers.LSTM(lstm_units[2], return_sequences=True, activation='gelu',
    name=f"{name_prefix}_decoder_lstm_1")(bottleneck_lstm)
    
    decoder_lstm_2 = layers.LSTM(lstm_units[3], return_sequences=True, activation='gelu',
    name=f"{name_prefix}_decoder_lstm_2")(decoder_lstm_1)
    reconstruction = layers.Flatten(name=f"{name_prefix}_reconstruction_flatten")(decoder_lstm_2)
    flattened_bottleneck = layers.Flatten(name=f"{name_prefix}_bottleneck_flatten")(bottleneck_lstm)
    dense_1 = layers.Dense(dense_units[0], activation='gelu', name=f"{name_prefix}_dense_1")(flattened_bottleneck)
    dense_2 = layers.Dense(dense_units[1], activation='gelu', name=f"{name_prefix}_dense_2")(dense_1)
    prediction = layers.Dense(output_number, activation='relu', name=f"{name_prefix}_prediction_output")(dense_2)
    
    return prediction, reconstruction

def create_final_branch(input_tensor, output_number, name_prefix):
    reshaped_input = layers.Reshape((1,672), name=f"{name_prefix}_reshape")(input_tensor)

    lstm_1 = layers.LSTM(168, return_sequences=True, activation='gelu', name=f"{name_prefix}_lstm_1")(reshaped_input)
    lstm_2 = layers.LSTM(42, return_sequences=True, activation='gelu', name=f"{name_prefix}_lstm_2")(lstm_1)
    flattened = layers.Flatten(name=f"{name_prefix}_flatten")(lstm_2)
    dense_1 = layers.Dense(48, activation='gelu', name=f"{name_prefix}_dense_1")(flattened)
    dense_2 = layers.Dense(24, activation='gelu')(dense_1)
    prediction = layers.Dense(output_number, activation='gelu', name=f"{name_prefix}_prediction_output")(dense_2)
    return prediction

##############################################################
##############################################################



# --- 使用示例 ---
if __name__ == '__main__':
    # 现在您可以轻松地构建模型并查看其摘要
    my_model = LSTM_Mixer(output_number=1)  # 假设最终需要分成 5 类
    my_model.summary()
    from keras.utils import plot_model
    plot_model(my_model, to_file='.\model_auth.png', show_shapes=True)

