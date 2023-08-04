

from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.backend import softmax
 
# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
 
    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
 
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
 
        # Computing the weights by a softmax operation
        weights = softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)
    
class MultiHeadAttention(Layer):
    def __init__(self,num_heads, key_dim, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = num_heads  # Number of attention heads to use
        self.d_k = key_dim  # Dimensionality of the linearly projected queries and keys
        self.d_v = key_dim  # Dimensionality of the linearly projected values
        self.d_model = key_dim  # Dimensionality of the model
        self.W_q = Dense(key_dim)  # Learned projection matrix for the queries
        self.W_k = Dense(key_dim)  # Learned projection matrix for the keys
        self.W_v = Dense(key_dim)  # Learned projection matrix for the values
        self.W_o = Dense(key_dim)  # Learned projection matrix for the multi-head output
 
    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x
 
    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)


class SinePositionEncoding(tf.keras.layers.Layer):

    def __init__(
        self,
        max_wavelength=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.compute_dtype=tf.float32

    def call(self, inputs):
        # TODO(jbischof): replace `hidden_size` with`hidden_dim` for consistency
        # with other layers.
        if isinstance(inputs, tf.RaggedTensor):
            bounding_shape = inputs.bounding_shape()
            position_embeddings = (
                self._compute_trim_and_broadcast_position_embeddings(
                    bounding_shape,
                )
            )
            # then apply row lengths to recreate the same ragged shape as inputs
            return tf.RaggedTensor.from_tensor(
                position_embeddings,
                inputs.nested_row_lengths(),
            )
        else:
            return self._compute_trim_and_broadcast_position_embeddings(
                tf.shape(inputs),
            )

    def _compute_trim_and_broadcast_position_embeddings(self, shape):
        seq_length = shape[-2]
        hidden_size = shape[-1]
        position = tf.cast(tf.range(seq_length), self.compute_dtype)
        min_freq = tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
        timescales = tf.pow(
            min_freq,
            tf.cast(2 * (tf.range(hidden_size) // 2), self.compute_dtype)
            / tf.cast(hidden_size, self.compute_dtype),
        )
        angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(hidden_size) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
            tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )

        return tf.broadcast_to(positional_encodings, shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
            }
        )
        return config
    

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att =  MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 =  tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 =  tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 =  tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 =  tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs,inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,layer_num, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.blocks=[]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        for i in range(layer_num):
            self.blocks.append(TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate=0.1))

        self.position_encode=SinePositionEncoding()

    def call(self, input_tensor):
        x=self.position_encode(input_tensor)
        x=x+input_tensor
        x = self.layernorm(x)
        
        for block in self.blocks:
            x=block(x)
        return x

def get_sequence_model_transformer():


    input_depth =tf.keras.layers.Input((60, 512))
    input_video = tf.keras.layers.Input((60, 512))
    input_pose = tf.keras.layers.Input((60, 270))
    
    pose = tf.keras.layers.Dense(256, activation="relu")(input_pose)
    pose = tf.keras.layers.Dropout(0.4)(pose)
    pose = tf.keras.layers.Dense(512, activation="relu")(pose)

    depth  = tf.keras.layers.Dense(256, activation="relu")(input_depth)
    depth  = tf.keras.layers.Dropout(0.4)(depth)
    depth  = tf.keras.layers.Dense(512, activation="relu")(depth )
    
    video = tf.keras.layers.Dense(256, activation="relu")(input_video)
    video = tf.keras.layers.Dropout(0.4)(video)
    video = tf.keras.layers.Dense(512, activation="relu")(video)
    
    
    x = tf.keras.layers.Concatenate(axis=-1)([video,depth,pose])
    x_embed = tf.keras.layers.Dense(256)(x)

    x=TransformerEncoder(layer_num=2, embed_dim=256, num_heads=8, ff_dim=512, dropout_rate=0.1,)(x_embed)
    x =  tf.keras.layers.Flatten()(x)


    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    output = tf.keras.layers.Dense(226, activation="softmax")(x)

    rnn_model = CustomModel([input_video,input_depth,input_pose], output)
    rnn_model.summary()

    return rnn_model


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (input_pose) ,y = data

        with tf.GradientTape() as tape:
            y_pred = self([input_pose] ,training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def validation_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (input_pose) ,y = data

        y_pred = self([input_pose],training=False)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss =self.compute_loss(y=y, y_pred=y_pred)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}