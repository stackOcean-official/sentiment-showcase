import streamlit as st
from transformers import pipeline


@st.cache(allow_output_mutation=True, show_spinner=False)
def sentiment_analysis():
    model_name = "oliverguhr/german-sentiment-bert"
    sentiment_analysis_model = pipeline(model=model_name, tokenizer=model_name)
    return sentiment_analysis_model


# sidebar
st.sidebar.header("Anwendungsfälle")
first_case = st.sidebar.expander("Markenanalyse")
first_case.write(
    "Die meisten Menschen teilen ihre Meinung über soziale Medien wie Twitter"
    " oder Facebook. Es lohnt sich jedoch nicht für einen Menschen alle"
    " Kommentare durchzulesen und ein Meinungsbild zu erstellen. Mit einer"
    " Stimmungsanalyse kann dies sehr schnell und einfach erledigt werden."
    " Ebenso kann die Entwicklung über Zeit gemessen werden"
)
second_case = st.sidebar.expander("Kundensupport")
second_case.write(
    "Das Management von Kundensupportanfragen kann Herausforderungen mit sich"
    " bringen. Die richtige Priorisierung und Zuteilung können sich als"
    " schwierig erweisen. Dies ist mit einer Stimmungsanalyse insbesondere in"
    " Kombination mit einer Themenextraktion sehr einfach zu bewältigen."
)
third_case = st.sidebar.expander("Marktanalyse")
third_case.write(
    "Der Markt wandelt sich schneller als je zuvor. Um aktuell zu bleiben ist"
    " es wichtig herauszufinden was gerade gut ankommt oder welcher Konkurrent"
    " wie da steht. Dies kann man mit einer Stimmungsanalyse gut herausfinden."
    " Ebenso kann man den Erfolg von Markteting-Kampagnen messen"
)
fourth_case = st.sidebar.expander("Produktanalyse")
fourth_case.write(
    "Besonders gepaart mit anderen Daten wie z.B. demografischen "
)

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
