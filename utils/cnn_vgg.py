import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

def make_model(
        input_seq_len,
        conv_layers=2,
        conv_kernel_size=5,
        conv_activation='relu',
        conv_filters_first=128,
        conv_dropout=0.2,
        dense_units=[40],
        dense_activation=['relu'],
        dense_dropout=[0.2],
        n_outputs=1,
        output_activation='linear',
        layer_name_suffix='',
    ):

    # Input
    model_input = layers.Input(shape=(input_seq_len, 4))

    # Recurrent layers
    conv_output = model_input
    for conv_layer_idx in range(conv_layers):
        
        # Convolutional layer
        conv_layer_0 = layers.Conv1D(
            activation=conv_activation,
            padding='same',
            filters=conv_filters_first*(2**conv_layer_idx),
            kernel_size=conv_kernel_size,
            name=f'conv_{conv_layer_idx}_0{layer_name_suffix}',
        )

        conv_layer_1 = layers.Conv1D(
            activation=conv_activation,
            padding='same',
            filters=conv_filters_first*(2**conv_layer_idx),
            kernel_size=conv_kernel_size,
            name=f'conv_{conv_layer_idx}_1{layer_name_suffix}',
        )

        conv_pooling_layer = layers.MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same',
            name=f'conv_pooling_{conv_layer_idx}{layer_name_suffix}'
        )

        # Dropout
        conv_dropout_layer = layers.Dropout(
            conv_dropout,
            name=f'conv_dropout_{conv_layer_idx}{layer_name_suffix}',
        )
        
        # Layer output
        conv_output = conv_dropout_layer(conv_pooling_layer(conv_layer_1(conv_layer_0(conv_output))))

    # Flatten
    conv_flat_output = layers.Flatten()(conv_output)

    # Dense layers
    dense_output = conv_flat_output
    for dense_idx, dense_layer_units in enumerate(dense_units):
        
        dense_layer = layers.Dense(
            dense_layer_units,
            activation=dense_activation[dense_idx],
            name=f'dense_{dense_idx}{layer_name_suffix}',
        )
        
        dense_dropout_layer = layers.Dropout(
            dense_dropout[dense_idx],
            name=f'dense_dropout_{dense_idx}{layer_name_suffix}',
        )
        
        dense_output = dense_dropout_layer(dense_layer(dense_output))

    # Final layer
    output_layer = layers.Dense(
        n_outputs,
        name=f'dense_output{layer_name_suffix}',
        activation=output_activation,
    )
    model_output = output_layer(dense_output)

    model = models.Model(
        model_input,
        model_output,
    )

    return model

def load_model(model_path):
    return tensorflow.keras.models.load_model(model_path)
