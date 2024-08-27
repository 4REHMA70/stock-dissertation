import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Conv1D, GlobalAveragePooling1D, Dropout, MaxPooling1D, Permute, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tcn import TCN
from reservoirpy.nodes import Reservoir
from tensorflow.keras.models import Sequential
from sklearn.linear_model import LinearRegression

# Helper function for attention mechanism
def attention_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax')(a)
    a = Permute((2, 1), name='attention_vec')(a)
    outputs = Multiply()([inputs, a])
    return outputs

# ESN-LSTM
class ESN_LSTM(tf.keras.Model):
    def __init__(self, input_shape, reservoir_size=200, lstm_units=100, l2_reg=0.01, lr=0.3, sr=0.9):
        super(ESN_LSTM, self).__init__()
        self.input_shape = input_shape
        self.reservoir_size = reservoir_size
        self.lstm_units = lstm_units
        self.l2_reg = l2_reg

        # Create ReservoirPy ESN
        self.esn = Reservoir(
            units=reservoir_size,
            lr=lr,
            sr=sr,
            input_dim=input_shape[-1],
            seed=42
        )
        
        # Define the Keras layers
        self.lstm_layer = LSTM(
            lstm_units, 
            kernel_regularizer=l2(l2_reg)
        )
        self.dense_layer = Dense(
            1, 
            kernel_regularizer=l2(l2_reg)
        )

    def call(self, inputs):
        # Transform input through ESN
        x = tf.py_function(self._esn_transform, [inputs], tf.float32)
        x.set_shape([None, self.input_shape[0], self.reservoir_size])
        
        # Pass through LSTM and Dense layers
        x = self.lstm_layer(x)
        return self.dense_layer(x)

    def _esn_transform(self, x):
        # This method will be called by tf.py_function
        return np.array([self.esn.run(xi.numpy()) for xi in x])

    def fit(self, X, y, epochs=100, batch_size=32, validation_data=None, callbacks=None):
        self.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Prepare validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
        else:
            X_val, y_val = None, None

        # Train the model
        history = super().fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(X_val, y_val) if validation_data is not None else None,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, X):
        return super().predict(X)

# GRU-LSTM
def gru_lstm(input_shape, gru_units=64, lstm_units=50, dense_units=32, dropout_rate=0.3, l2_val=1e-4):
    model = Sequential()
    model.add(GRU(units=gru_units, return_sequences=True, input_shape=input_shape, 
                  kernel_regularizer=l2(l2_val), recurrent_regularizer=l2(l2_val)))
    model.add(LSTM(units=lstm_units, return_sequences=False, 
                   kernel_regularizer=l2(l2_val), recurrent_regularizer=l2(l2_val)))
    model.add(Dense(units=dense_units, activation='relu', kernel_regularizer=l2(l2_val)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Attention GRU-LSTM
def att_gru_lstm(input_shape, gru_units=64, lstm_units=50, dense_units=32, dropout_rate=0.3, l2_val=1e-4):
    inputs = Input(shape=input_shape)
    x = GRU(units=gru_units, return_sequences=True, kernel_regularizer=l2(l2_val), recurrent_regularizer=l2(l2_val))(inputs)
    x = LSTM(units=lstm_units, return_sequences=True, kernel_regularizer=l2(l2_val), recurrent_regularizer=l2(l2_val))(x)
    x = attention_block(x)
    x = Dense(units=dense_units, activation='relu', kernel_regularizer=l2(l2_val))(x)
    x = Dropout(dropout_rate)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = Dense(units=1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Attention LSTM-CNN
def att_lstm_cnn(input_shape, conv_filters=64, conv_kernel_size=3, lstm_units=50, dense_units=32, dropout_rate=0.3, l2_val=1e-4):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu', kernel_regularizer=l2(l2_val))(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = attention_block(x)
    x = LSTM(lstm_units, activation='relu', return_sequences=False, kernel_regularizer=l2(l2_val), recurrent_regularizer=l2(l2_val))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, kernel_regularizer=l2(l2_val))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model

# TCN
def tcn(input_shape, output_size=1, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8, 16], dropout_rate=0.2, learning_rate=0.001):
    inputs = Input(shape=input_shape)
    x = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations,
            padding='causal', use_skip_connections=True, dropout_rate=dropout_rate, return_sequences=False)(inputs)
    x = Dense(48, activation='relu', kernel_regularizer=l2(1e-3))(x)
    outputs = Dense(output_size)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# DeepESN
class DeepESN:
    def __init__(self, n_inputs, n_reservoir=200, n_outputs=1, n_layers=3, 
                 spectral_radius=0.9, sparsity=0, leaking_rate=1.0, 
                 ridge_param=1e-3, random_state=None):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.ridge_param = ridge_param
        self.random_state = random_state
        
        np.random.seed(self.random_state)
        
        self.W_in = [np.random.rand(n_reservoir, n_inputs) - 0.5]
        self.W_in += [np.random.rand(n_reservoir, n_reservoir) - 0.5 for _ in range(n_layers - 1)]
        
        self.W = [np.random.rand(n_reservoir, n_reservoir) - 0.5 for _ in range(n_layers)]
        
        self.W_out = None
        
        for l in range(n_layers):
            if self.sparsity > 0:
                mask = np.random.rand(*self.W[l].shape) < self.sparsity
                self.W[l][mask] = 0
            
            radius = np.max(np.abs(np.linalg.eigvals(self.W[l])))
            self.W[l] *= (self.spectral_radius / radius)

    def _update(self, state, input, layer):
        new_state = np.tanh(np.dot(self.W_in[layer], input) + np.dot(self.W[layer], state))
        return (1 - self.leaking_rate) * state + self.leaking_rate * new_state

    def fit(self, X, y, warmup=0):
        n_samples, n_timesteps, n_features = X.shape
        states = [np.zeros((n_samples, n_timesteps, self.n_reservoir)) for _ in range(self.n_layers)]
        
        for i in range(n_samples):
            layer_input = X[i]
            for l in range(self.n_layers):
                for t in range(n_timesteps):
                    if t > 0:
                        states[l][i, t] = self._update(states[l][i, t-1], layer_input[t], l)
                    else:
                        states[l][i, t] = self._update(np.zeros(self.n_reservoir), layer_input[t], l)
                layer_input = states[l][i]

        final_states = [s[:, -1, :] for s in states]
        extended_states = np.hstack(final_states + [np.ones((n_samples, 1))])

        identity = np.eye(extended_states.shape[1])
        self.W_out = np.linalg.inv(extended_states.T.dot(extended_states) + 
                                   self.ridge_param * identity).dot(extended_states.T).dot(y)

    def predict(self, X):
        n_samples, n_timesteps, n_features = X.shape
        states = [np.zeros((n_samples, n_timesteps, self.n_reservoir)) for _ in range(self.n_layers)]
        
        for i in range(n_samples):
            layer_input = X[i]
            for l in range(self.n_layers):
                for t in range(n_timesteps):
                    if t > 0:
                        states[l][i, t] = self._update(states[l][i, t-1], layer_input[t], l)
                    else:
                        states[l][i, t] = self._update(np.zeros(self.n_reservoir), layer_input[t], l)
                layer_input = states[l][i]

        final_states = [s[:, -1, :] for s in states]
        extended_states = np.hstack(final_states + [np.ones((n_samples, 1))])

        y_pred = np.dot(extended_states, self.W_out)
        return y_pred

# Ensemble Model
class EnsembleModel:
    def __init__(self, input_shape, n_features, deep_esn_params=None, tcn_params=None, att_lstm_cnn_params=None):
        self.input_shape = input_shape
        self.n_features = n_features
        
        deep_esn_params = deep_esn_params or {}
        self.deep_esn = DeepESN(n_inputs=n_features, **deep_esn_params)
        
        tcn_params = tcn_params or {}
        self.tcn = tcn(input_shape, **tcn_params)
        
        att_lstm_cnn_params = att_lstm_cnn_params or {}
        self.attention_lstm = att_lstm_cnn(input_shape, **att_lstm_cnn_params)
        
    def fit(self, X, y, validation_data, epochs=200, batch_size=32, callbacks=None):
        X_val, y_val = validation_data
        # Reshape X for DeepESN if necessary
        X_esn = X.reshape(X.shape[0], X.shape[1], -1)
        X_val_esn = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)
        
        # Fit DeepESN
        self.deep_esn.fit(X_esn, y)
        
        # Fit TCN and Attention-LSTM
        self.tcn.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        self.attention_lstm.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        
    def predict(self, X):
        X_esn = X.reshape(X.shape[0], X.shape[1], -1)
        pred_esn = self.deep_esn.predict(X_esn).flatten()
        pred_tcn = self.tcn.predict(X).flatten()
        pred_att_lstm = self.attention_lstm.predict(X).flatten()
        ensemble_pred = (pred_esn + pred_tcn + pred_att_lstm) / 3
        return ensemble_pred.reshape(-1, 1)

class StackingEnsembleModel(EnsembleModel):
    def __init__(self, input_shape, n_features, deep_esn_params=None, tcn_params=None, att_lstm_cnn_params=None):
        super().__init__(input_shape, n_features, deep_esn_params, tcn_params, att_lstm_cnn_params)
        self.meta_model = LinearRegression()

    def fit(self, X, y, validation_data, epochs=200, batch_size=32, callbacks=None):
        X_val, y_val = validation_data
        super().fit(X, y, (X_val, y_val), epochs, batch_size, callbacks)
        
        # Generate base model predictions for training meta-model
        X_meta = np.column_stack([
            self.deep_esn.predict(X.reshape(X.shape[0], X.shape[1], -1)).flatten(),
            self.tcn.predict(X).flatten(),
            self.attention_lstm.predict(X).flatten()
        ])
        
        # Fit meta-model
        self.meta_model.fit(X_meta, y)

    def predict(self, X):
        X_esn = X.reshape(X.shape[0], X.shape[1], -1)
        pred_esn = self