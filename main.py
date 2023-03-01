# importing libraries
import streamlit as st
import spacy
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests


nlp = spacy.load("en_core_web_sm")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():

    # sidebar for navigation
    with st.sidebar:
        
        selected = option_menu('Named Entity Recognition',
                            
                            ['Home','Entity Recognizer'],
                            icons=['activity','heart',],
                            default_index=0)

    if selected == "Home":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Named Entity Recognition</h1>", unsafe_allow_html=True)
        st.markdown("<h4 '>A web app to recognize the enitities in a given text</h4>", unsafe_allow_html=True)
        lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_1plcwvk5.json")
        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )
        st.markdown("<h4 '>About:</h4>", unsafe_allow_html=True)
        st.write("Named Entity Recognition (NER) is a natural language processing technique that involves identifying and categorizing named entities in unstructured text data. Named entities can be people, places, organizations, dates, times, and more. NER is an important component of many text analysis tasks, including information retrieval, information extraction, and text classification. It is widely used in fields such as finance, healthcare, and social media analysis")

        st.markdown("<h4 '>Features:</h4>", unsafe_allow_html=True)
        st.write("Easy Detection of diseases in leafs: Just need to click and upload leaf image.")
        st.write("Fast and Accurate: Provides the disease with high accuracy and fast")
        st.write("Cause and Solution of diseases: Provides the cause and solution of the disease")
        st.write('Large Plant Support: Supports 38 different plants')

        # list of plants we can detect
        st.markdown("<h4  '>List of types we can identify</h4>", unsafe_allow_html=True)
        st.write(nlp.pipe_labels['ner'])
        lotti = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_srcvuh0h.json")
        st_lottie(
            lotti,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )

    elif selected == "Entity Recognizer":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Entity Recognizer</h1>", unsafe_allow_html=True)
        lot = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_0xbu1xfo.json')
        st_lottie(
            lot,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )
        text = st.text_area("Enter the text to recognize entities")
        if st.button("Recognize"):
            doc = nlp(text)
            st.write([(ent.text, ent.label_) for ent in doc.ents])

if __name__ == "__main__":
    main()
