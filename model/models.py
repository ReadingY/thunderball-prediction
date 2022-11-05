import keras
from keras import layers, models, optimizers, losses
import settings

inputs = []
outputs = []
front_temps = []
back_temps = []

for i in range(settings.FRONT_SIZE):
    # Input layer
    x_input = layers.Input((settings.MAX_STEPS, settings.FRONT_NUMBERS_RANGE), name='x{}'.format(i + 1))

    # Neural network
    x = layers.Bidirectional(layers.LSTM(settings.LSTM_UNITS, return_sequences=True))(x_input)

    # Config dropout rate
    x = layers.Dropout(rate=settings.DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.LSTM(settings.LSTM_UNITS, return_sequences=True))(x)
    x = layers.Dropout(rate=settings.DROPOUT_RATE)(x)
    x = layers.TimeDistributed(layers.Dense(settings.FRONT_NUMBERS_RANGE * 3))(x)

    # Flat network
    x = layers.Flatten()(x)

    # Connect all layers
    x = layers.Dense(settings.FRONT_NUMBERS_RANGE * 3, activation='relu')(x)

    inputs.append(x_input)

    front_temps.append(x)

for i in range(settings.BACK_SIZE):
    # Input layer
    x_input = layers.Input((settings.MAX_STEPS, settings.BACK_THUNDERBALL_RANGE),
                           name='x{}'.format(i + 1 + settings.FRONT_SIZE))

    # Neural network
    x = layers.Bidirectional(layers.LSTM(settings.LSTM_UNITS, return_sequences=True))(x_input)

    # Config dropout rate
    x = layers.Dropout(rate=settings.DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.LSTM(settings.LSTM_UNITS, return_sequences=True))(x)
    x = layers.Dropout(rate=settings.DROPOUT_RATE)(x)
    x = layers.TimeDistributed(layers.Dense(settings.BACK_THUNDERBALL_RANGE * 3))(x)

    # Flat network
    x = layers.Flatten()(x)

    # Connect all layers
    x = layers.Dense(settings.BACK_THUNDERBALL_RANGE * 3, activation='relu')(x)

    inputs.append(x_input)

    back_temps.append(x)

front_concat_layer = layers.concatenate(front_temps)
back_concat_layer = layers.concatenate(back_temps)

# Use softmax to compute distribution
for i in range(settings.FRONT_SIZE):
    x = layers.Dense(
        settings.FRONT_NUMBERS_RANGE,
        activation='softmax', name='y{}'.format(i + 1))(front_concat_layer)
    outputs.append(x)

for i in range(settings.BACK_SIZE):
    x = layers.Dense(
        settings.BACK_THUNDERBALL_RANGE,
        activation='softmax',
        name='y{}'.format(i + 1 + settings.FRONT_SIZE))(back_concat_layer)
    outputs.append(x)

# Create model
model = models.Model(inputs, outputs)

# Specify optimizer and loss functions
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.categorical_crossentropy for __ in range(settings.FRONT_SIZE + settings.BACK_SIZE)],
    loss_weights=[2, 2, 2, 2, 2, 1])

model.summary()
