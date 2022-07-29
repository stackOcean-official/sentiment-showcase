# sentiment-showcase

Sample Deployment of a Sentiment Analysis Model with Streamlit UI using Docker and HuggingFace

---

## How to setup

Create a new venv (virtual environment):

```
python3 -m venv .venv
```

Activate new environment:

For Mac/Linux:

```
source .venv/bin/activate
```

For Windows:

```
source .venv/Scripts/activate
```

Install packages:

```
pip install -r requirements.txt
```

## Run the streamlit app

```
python3 -m streamlit run src/app.py
```

---

## How to build & run the Docker Image

```
docker build -t sentiment-showcase .
```

Run the Docker Image with:

```
docker run -d --rm -p 8501:8501 --name sentiment-showcase sentiment-showcase
```

You can now access the streamlit server at [http://localhost:8501](http://localhost:8501)

<br/>

To view the logs of the container run:

```
docker logs sentiment-showcase
```

To stop the container run:

```
docker stop sentiment-showcase
```
