from tensorflow import keras

def build_model(input_shape, output_length, n_layers, width, hidden_activation, final_activation):

    input_layer = keras.Input(shape=input_shape, name='Input-layer')
    x = input_layer

    for i in range(n_layers):
        x = keras.layers.Dense(units=width, activation=hidden_activation, name=f'Dense-layer-{i}-{width}-units')(x)

    output = keras.layers.Dense(output_length, activation=final_activation, name='Ouput-layer')(x)
    model = keras.Model(input_layer, output, name='Model')

    return model
