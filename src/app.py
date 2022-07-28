import streamlit as st
from transformers import pipeline


@st.cache(allow_output_mutation=True, show_spinner=False)
def sentiment_analysis():
    model_name = "oliverguhr/german-sentiment-bert"
    sentiment_analysis_model = pipeline(model=model_name, tokenizer=model_name)
    return sentiment_analysis_model


# main page
st.title("Stimmungsanalyse")
explanation = st.expander("Wie funktioniert es?")
explanation.write(
    "Bei einer Stimmungsanalyse analysiert das Modell ob ein Satz positiv,"
    " neutral oder negativ gemeint ist. Dazu liefert es einen prozentualen"
    " Wert, der aussagt, wie sicher sich das Modell bei seinem Ergebnis ist."
)
what_else = st.expander("Was kann man noch machen?")
explanation.write(
    "Weitere Schritte, die man angehen könnte, wären, dass man das Thema des"
    " Satzes mit herausfiltert (Themenextraktion) oder die genaue Emotion"
    " erkennt."
)


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
