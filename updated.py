import pandas as pd
import numpy as np
import streamlit as st
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

tags = ['I-artifact',
 'I-person',
 'B-natural Phenomenon',
 '--',
 'B-event',
 'I-geographic entity',
 'B-artifact',
 'I-natural Phenomenon',
 'B-person',
 'B-geographic entity',
 'I-event',
 'B-time indicator',
 'B-geopolitical entity',
 'B-organization',
 'I-time indicator',
 'I-geopolitical entity',
 'I-organization']

word_idx = json.load(open('word_idx.json', 'r'))
model = keras.models.load_model('model.h5')

def tag_sentence(text:str):
    word_list = text.split(" ")
    x_new = []
    for word in word_list:
        if word in word_idx:
            x_new.append(word_idx[word])
        else:
            x_new.append(0)

    x_new = pad_sequences([x_new], maxlen=50, padding='post')
    y_pred = model.predict(x_new)
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = [tags[i] for i in y_pred[0]]
    st.write("-" * 35)
    word1 = []
    pred1 = []
    for word, pred in zip(word_list, y_pred):
        word1.append(word)
        pred1.append(pred)

    df = pd.DataFrame({'Word':word1, 'Prediction':pred1})
    st.write(df)

def main():
    st.title("Named Entity Recognition")
    text = st.text_area("Enter Text")
    if st.button("Tag"):
        tag_sentence(text)

if __name__ == '__main__':
    main()