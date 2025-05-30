import tensorflow as tf

def build_simple_rnn_model(vocab_size, embedding_dim=128, rnn_units=64, num_classes=3,
                           sequence_length=100, bidirectional=False, num_layers=1,dropout_rate=0.5):
    inputs = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name = "input_layer")
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name = "embedding")(inputs)

    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)  
        rnn = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=return_sequences, name=f"rnn_layer_{i+1}")
        if bidirectional:
            x = tf.keras.layers.Bidirectional(rnn, name=f"bidirectional_{i+1}")(x)
        else:
            x = rnn(x)

    x = tf.keras.layers.Dropout(0.5, name= "dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name = "output")(x)

    return tf.keras.Model(inputs, outputs)

