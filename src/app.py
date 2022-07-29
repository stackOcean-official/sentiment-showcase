import streamlit as st
from transformers import pipeline


@st.cache(allow_output_mutation=True, show_spinner=False)
def sentiment_analysis():
    model_name = "oliverguhr/german-sentiment-bert"
    sentiment_analysis_model = pipeline(model=model_name, tokenizer=model_name)
    return sentiment_analysis_model


sample_input = (
    "Erst hat die Lieferung ewig gedauert und dann war es auch noch das"
    " falsche Produkt. Großartig!"
)
input_text = st.text_area(
    label=(
        "Schreibe einen deutschen Satz, oder versuche es mit dem vorgegebenen"
        " Text"
    ),
    value=sample_input,
    height=80,
    max_chars=128,
)
compute = st.button("Analysiere Text")
if compute:
    with st.spinner("Lade Modell und berechne..."):
        model = sentiment_analysis()
        output = model(input_text)
        if output[0]["label"] == "neutral":
            output_label = "Neutral"
        else:
            output_label = output[0]["label"][:-1].capitalize()
        st.subheader(
            f"Ergebnis: {output_label}, Zuversicht:"
            f" {round(output[0]['score'] * 100, 2)}%"
        )


# remove menu for production
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
