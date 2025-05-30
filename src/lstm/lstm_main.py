import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics
from sklearn.metrics import f1_score
from typing import Tuple, List
from lstm import (
    load_lstm_weights_from_keras,
    create_lstm_model_from_weights,
    compute_macro_f1_score
)

def load_dataset(base_path: str
    ) -> Tuple[Tuple[List[str], List[int]],
               Tuple[List[str], List[int]],
               Tuple[List[str], List[int]]]:

    df_train = pd.read_csv(os.path.join(base_path, 'train.csv'))
    df_val   = pd.read_csv(os.path.join(base_path, 'valid.csv'))
    df_test  = pd.read_csv(os.path.join(base_path, 'test.csv'))


    unique_labels = sorted(df_train['label'].unique())
    label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    print("Label mapping:", label2idx)  

    def _to_xy(df: pd.DataFrame):
        texts = df['text'].astype(str).tolist()
        labels = df['label'].map(label2idx).astype(int).tolist()
        return texts, labels

    return _to_xy(df_train), _to_xy(df_val), _to_xy(df_test)

def build_vectorizer(max_tokens=20000, seq_len=200):
    return layers.TextVectorization(max_tokens=max_tokens,
                                    output_mode='int',
                                    output_sequence_length=seq_len)

def make_tf_dataset(texts, labels, vectorizer, batch_size=64, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=42)
    ds = ds.batch(batch_size).map(lambda x, y: (vectorizer(x), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def build_keras_lstm(vocab_size, embedding_dim, seq_len,
                     lstm_units, bidirectional, dropout_rate,
                     num_classes):
    inp = layers.Input((seq_len,), dtype='int32')
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=False)(inp)
    for i, u in enumerate(lstm_units):
        ret_seq = (i < len(lstm_units)-1)
        L = layers.LSTM(u, return_sequences=ret_seq)
        if bidirectional:
            L = layers.Bidirectional(L)
        x = L(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m = models.Model(inp, out)
    m.compile(optimizer=optimizers.Adam(),
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=[metrics.SparseCategoricalAccuracy()])
    return m

def eval_macro_f1_keras(model, ds):
    y_true, y_pred = [], []
    for x, y in ds:
        p = np.argmax(model.predict(x), axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(p.tolist())
    return f1_score(y_true, y_pred, average='macro')

def main():
    DATA_DIR    = 'dataset'
    MODEL_DIR   = 'models'
    MAX_TOKENS  = 20000
    SEQ_LEN     = 200
    EMBED_DIM   = 128
    LSTM_UNITS  = [64]       
    BIDIR       = True
    DROPOUT     = 0.5
    BATCH_SIZE  = 64
    EPOCHS      = 10        

    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_dataset(DATA_DIR)

    vectorizer = build_vectorizer(MAX_TOKENS, SEQ_LEN)
    vectorizer.adapt(train_texts)

    train_ds = make_tf_dataset(train_texts, train_labels, vectorizer, BATCH_SIZE, shuffle=True)
    val_ds   = make_tf_dataset(val_texts,   val_labels,   vectorizer, BATCH_SIZE)
    test_ds  = make_tf_dataset(test_texts,  test_labels,  vectorizer, BATCH_SIZE)

    os.makedirs(MODEL_DIR, exist_ok=True)
    keras_model = build_keras_lstm(
        vocab_size=MAX_TOKENS,
        embedding_dim=EMBED_DIM,
        seq_len=SEQ_LEN,
        lstm_units=LSTM_UNITS,
        bidirectional=BIDIR,
        dropout_rate=DROPOUT,
        num_classes=len(set(train_labels))
    )
    keras_model.summary()

    ckpt = callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'lstm_full.h5'),
        save_best_only=True, monitor='val_loss'
    )
    es = callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    keras_model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=[ckpt, es])

    acc_keras = keras_model.evaluate(test_ds, verbose=0)[1]
    f1_keras  = eval_macro_f1_keras(keras_model, test_ds)
    print(f"\n--- Keras LSTM Results ---")
    print(f"Test Accuracy:           {acc_keras:.4f}")
    print(f"Test Macro-F1 (Keras):   {f1_keras:.4f}")

    h5_path = os.path.join(MODEL_DIR, 'lstm_full.h5')
    weights = load_lstm_weights_from_keras(
        h5_file_path=h5_path,
        bidirectional=BIDIR,
        num_lstm_layers=len(LSTM_UNITS)
    )
    scratch_model = create_lstm_model_from_weights(weights, dropout_rate=DROPOUT,bidirectional=BIDIR)

    test_seq = vectorizer(np.array(test_texts)).numpy()  

    preds_proba = scratch_model.predict(test_seq)
    preds_label = np.argmax(preds_proba, axis=1)
    acc_scratch = np.mean(preds_label == np.array(test_labels))
    f1_scratch  = compute_macro_f1_score(np.array(test_labels), preds_label)

    print(f"\n--- Scratch LSTM Results ---")
    print(f"Test Accuracy (scratch): {acc_scratch:.4f}")
    print(f"Test Macro-F1 (scratch): {f1_scratch:.4f}")

if __name__ == '__main__':
    main()