
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense, Concatenate, Dropout, GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention, Add, Layer
)

# Transformer encoder block with optional tweaks: residual and normalization
def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0):
    # Multi-head self-attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    # Feed-forward layer
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

# Main input: 40x14 matrix
main_input = Input(shape=(40, 14), name='main_input')
x_main = transformer_encoder(main_input)
x_main = GlobalAveragePooling1D()(x_main)
x_main = Dense(560, activation='relu')(x_main)
x_main = Dense(280, activation='relu')(x_main)
x_main = Dense(140, activation='relu')(x_main)

# Side inputs (with halving layers + residual + concatenate original features)
side_inputs = []
side_processed = []
side_shapes = [4, 6, 5, 5, 8, 3, 3]

for i, shape in enumerate(side_shapes):
    input_i = Input(shape=(shape,), name=f'side_input_{i+1}')
    d1 = Dense(shape, activation='relu')(input_i)
    d2 = Dense(max(1, shape // 2), activation='relu')(d1)
    # Optional: concatenate original + processed for richer representation
    d_combined = Concatenate()([input_i, d2])
    side_inputs.append(input_i)
    side_processed.append(d_combined)

# Concatenate all side features
x_side = Concatenate()(side_processed)
x_side = Dense(14, activation='relu')(x_side)  # Adjusted for richer fusion

# Merge main and side
x = Concatenate()([x_main, x_side])
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(4, activation='relu')(x)
output = Dense(1, activation='sigmoid', name='output')(x)

# Build and compile model
model = Model(inputs=[main_input] + side_inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


