import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

# Custom attention layer
class BahdanauAttention1D(layers.Layer):
    def __init__(self, units, name=None, **kwargs):
        super(BahdanauAttention1D, self).__init__(name=name, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='W',
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
            name='b',
        )
        self.v = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
            name='v',
        )

    def call(self, inputs, mask=None):
        # Mask, size: [batch_size, Tv]
        if mask is None:
            mask_float = 1.
            mask_bc = 1.
        else:
            mask_float = tensorflow.cast(mask, "float32")
            # Broadcasted mask, size: [batch_size, Tv, 1]
            mask_bc = tensorflow.expand_dims(mask_float, -1)
        # Query: vector v, size [units]
        query = self.b
        # Values: inputs*W, size [batch_size, Tv, units]
        values = tensorflow.matmul(inputs*mask_bc, self.W)
        # Score: v dot tanh(query + values), size [batch_size, Tv]
        score = tensorflow.reduce_sum(
            self.v * tensorflow.nn.tanh(query + values) * mask_bc,
            axis=-1,
        )
        # Weights: softmax across time axis
        # Size [batch_size, Tv]
        # weights = tensorflow.nn.softmax(score, axis=1)
        score_exp = tensorflow.exp(score) * mask_float
        score_sum = tensorflow.reduce_sum(
            score_exp * mask_float, axis=-1, keepdims=True
        )
        weights = score_exp / score_sum
        # Output: sum (weights * values) across time axis
        # Shape: [batch_size, units]
        output = tensorflow.reduce_sum(
            tensorflow.expand_dims(weights, 2) * values * mask_bc,
            axis=1,
        )
        
        return output, weights
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config

def make_model(
        rnn_type='gru',
        rnn_units=[321],
        rnn_dropout=0.0,
        attn_units=151,
        dense_units=[1000],
        dense_dropout=0.0,
        n_outputs=8,
        output_activation='linear',
        layer_name_suffix='',
    ):

    # Input
    model_input = layers.Input(shape=(None, 4))

    # Mask layer
    masked_input = layers.Masking(mask_value=-1, name=f'input_mask{layer_name_suffix}')(model_input)

    # Recurrent layers
    rnn_output = masked_input
    for rnn_idx, rnn_layer_units in enumerate(rnn_units):
        
        # Recurrent layer type
        if rnn_type == 'lstm':
            rnn_layer = layers.LSTM(rnn_layer_units, return_sequences=True)
        elif rnn_type == 'gru':
            rnn_layer = layers.GRU(rnn_layer_units, return_sequences=True)
        else:
            raise ValueError('rnn_type {} not recognized'.format(rnn_type))
        
        # Bidirectional layer from recurrent layer
        bi_rnn_layer = layers.Bidirectional(
                rnn_layer,
                name=f'rnn_bi_{rnn_idx}{layer_name_suffix}',
            )
        
        # Dropout
        rnn_dropout_layer = layers.Dropout(rnn_dropout, name=f'rnn_dropout_{rnn_idx}{layer_name_suffix}')
        
        # Layer output
        rnn_output = rnn_dropout_layer(bi_rnn_layer(rnn_output))

    # Attention layer
    attn_layer = BahdanauAttention1D(attn_units, name=f'attention{layer_name_suffix}')
    attn_output, attn_weights = attn_layer(rnn_output)

    # Dense layers
    dense_output = attn_output
    for dense_idx, dense_layer_units in enumerate(dense_units):
        
        dense_layer = layers.Dense(
            dense_layer_units,
            name=f'dense_{dense_idx}{layer_name_suffix}',
            activation='relu',
        )
        
        dense_dropout_layer = layers.Dropout(dense_dropout, name=f'dense_dropout_{dense_idx}{layer_name_suffix}')
        
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
    model = tensorflow.keras.models.load_model(
        model_path,
        custom_objects={'BahdanauAttention1D': BahdanauAttention1D},
    )

    return model
