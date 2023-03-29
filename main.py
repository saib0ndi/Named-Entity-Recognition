# importing libraries
import streamlit as st
import spacy
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from spacy.cli import download

nlp = spacy.load("en_core_web_sm")

def main():
    
    st.markdown("<h1 style='text-align: center; color: #08e08a;'>Entity Recognizer</h1>", unsafe_allow_html=True)
    text = st.text_area("Enter the text to recognize entities")
    if st.button("Recognize"):
        
        doc = nlp(text)
        htnl = spacy.displacy.render(doc, style="ent")
        st.write(htnl, unsafe_allow_html=True)

if __name__ == "__main__":
    main()