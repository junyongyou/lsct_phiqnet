import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Layer


def create_padding_mask(input):
    """
    Creates mask for input to Transformer based on the average of all elements = 0
    :param input: input sequence
    :return: mask
    """
    input = tf.pad(input, paddings=[[0, 0], [1, 0], [0, 0]], constant_values=1)
    input = tf.cast(tf.math.equal(tf.keras.backend.mean(input, axis=-1), 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return input[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class MultiHeadAttention(Layer):
    """
    This is the standard multi-head attention layer
    """
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(
                f'embedding dimension = {d_model} should be divisible by number of heads = {num_heads}'
            )
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = matmul_qk / tf.math.sqrt(dim_key)
        if mask is not None:
            scaled_score += (mask * -1e9)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def call(self, inputs, mask):
        batch_size = tf.shape(inputs)[0]

        query = self.wq(inputs)
        key = self.wk(inputs)
        value = self.wv(inputs)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention, weights = self.scaled_dot_product_attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.d_model)
        )
        output = self.dense(concat_attention)
        return output, weights


class TransformerBlock(Layer):
    """
    This is the standard Transformer block
    """
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(dff, activation="relu"),
             Dense(d_model),]
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x, training, mask):
        attn_output, attention_weigths = self.mha(x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class VideoQualityTransformer(Model):
    """
    Transformer for video quality assessment using the standard Transformer,
    the maximum_position_encoding should cover the maximal clip number in the databases
    """
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        mlp_dim,
        dropout=0.1,
        maximum_position_encoding=6000
    ):
        super(VideoQualityTransformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # positional embedding is predefined with a sufficient length
        self.pos_emb = self.add_weight('pos_emb', shape=(1, maximum_position_encoding, d_model))

        # add video quality token
        self.quality_emb = self.add_weight('quality_emb', shape=(1, 1, d_model))

        # normal Transformer architecture
        self.feature_proj = Dense(d_model)
        self.dropout = Dropout(dropout)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]

        # MLP head
        self.mlp_head = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                Dense(1),
            ]
        )

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        mask = create_padding_mask(x)

        frame_length = tf.shape(x)[1]
        x = self.feature_proj(x)

        quality_emb = tf.broadcast_to(self.quality_emb, [batch_size, 1, self.d_model])
        x = tf.concat([quality_emb, x], axis=1)

        # truncate the positional embedding for shorter videos
        x = x + self.pos_emb[:, : frame_length + 1, :]

        x = self.dropout(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, training, mask)

        # First (CLS) is used for VQA
        x = self.mlp_head(x[:, 0])
        return x