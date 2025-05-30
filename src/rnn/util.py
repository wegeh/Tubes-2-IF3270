import tensorflow as tf

from keras_model import build_simple_rnn_model

SEED = 42
def prepare_dataset(texts, labels, vectorizer, batch_size=32):
    texts = tf.convert_to_tensor(texts, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    ds = ds.map(lambda x, y: (vectorizer(x), y))
    ds = ds.shuffle(1024, seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train_rnn_model(
    train_ds, 
    val_ds,
    vocab_size,
    embedding_dim,
    rnn_units,
    num_classes,
    sequence_length,
    bidirectional,
    num_layers,
    epochs=20,
    patience=3,
    verbose=1
):
    print("Membangun model...")
    model = build_simple_rnn_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        num_classes=num_classes,
        sequence_length=sequence_length,
        bidirectional=bidirectional,
        num_layers=num_layers
    )
    
    # Menampilkan ringkasan arsitektur model
    model.summary(line_length=100)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # 3. Mendefinisikan callback untuk Early Stopping
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # 4. Melatih model
    print(f"\nMemulai training model untuk {epochs} epoch (dengan Early Stopping patience={patience})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[callback],
        verbose=verbose 
    )
    
    print("\nTraining selesai.")
    
    return model, history