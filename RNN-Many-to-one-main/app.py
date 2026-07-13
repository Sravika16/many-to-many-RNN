# RNN Practical (Many to One)
# Dataset: spam.csv

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    Dropout,
    Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


# ======================================================
# Configuration
# ======================================================

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50


# ======================================================
# Clean Text
# ======================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================================================
# Train Model
# ======================================================

def train_model():

    print("Training dataset...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print(df.head())
    print(df["label"].value_counts())

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    df["text"] = df["text"].apply(clean_text)

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    y = df["label"].values

    print("X Shape :", X.shape)
    print("y Shape :", y.shape)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Class Weights
    # --------------------------------------------------

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    print("\nClass Weights:")
    print(class_weights)

    # --------------------------------------------------
    # Build Model
    # --------------------------------------------------

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=64,
            input_length=MAX_LEN
        )
    )

    model.add(
        Bidirectional(
            SimpleRNN(64)
        )
    )

    model.add(
        Dropout(0.5)
    )

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.build(input_shape=(None, MAX_LEN))

    model.summary()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test),
        class_weight=class_weights,
        verbose=1
    )

    model.save(MODEL)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print("\nTest Loss :", loss)
    print("Accuracy :", accuracy)

    probabilities = model.predict(
        x_test,
        verbose=0
    )

    predictions = (probabilities > 0.5).astype(int).flatten()

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


# ======================================================
# Predict
# ======================================================

def predict_sms(message):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences(
        [message]
    )

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = float(model.predict(sequence, verbose=0)[0][0])
    print("Prediction Probability:", probability)

    if probability >= 0.5:
        return "🚨 Spam", probability

    return "✅ Ham", 1 - probability


# ======================================================
# Train Once
# ======================================================

if not os.path.exists(MODEL):
    train_model()


# ======================================================
# Streamlit UI
# ======================================================

st.set_page_config(
    page_title="Spam Detection",
    page_icon="📩"
)

st.title("📩 Spam Detection using RNN")

st.write(
    "Enter an SMS message and the trained RNN model will "
    "predict whether it is **Spam** or **Ham**."
)

message = st.text_area(
    "Enter Message"
)

if st.button("Predict"):

    if message.strip():

        prediction, confidence = predict_sms(message)

        if "Spam" in prediction:
            st.error(prediction)
        else:
            st.success(prediction)

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

    else:
        st.warning(
            "Please enter a message."
        )