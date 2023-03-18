import pickle
import pandas as pd
import numpy as np
import streamlit as st

id2tag = {0: 'O',
 1: 'B-corporation',
 2: 'I-corporation',
 3: 'B-creative-work',
 4: 'I-creative-work',
 5: 'B-group',
 6: 'I-group',
 7: 'B-location',
 8: 'I-location',
 9: 'B-person',
 10: 'I-person',
 11: 'B-product',
 12: 'I-product'}

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def tag_sentence(text:str):
    # convert our text to a  tokenized sequence
    inputs = tokenizer(text, truncation=True, return_tensors="pt").to("cuda")
    # get outputs
    outputs = model(**inputs)
    # convert to probabilities with softmax
    probs = outputs[0][0].softmax(1)
    # get the tags with the highest probability
    word_tags = [(tokenizer.decode(inputs['input_ids'][0][i].item()), id2tag[tagid.item()]) 
                  for i, tagid in enumerate (probs.argmax(axis=1))]

    return pd.DataFrame(word_tags, columns=['word', 'tag'])

def main():
    st.title("Named Entity Recognition")
    text = st.text_area("Enter Text")
    if st.button("Tag"):
        output = tag_sentence(text)
        st.table(output)

if __name__ == '__main__':
    main()