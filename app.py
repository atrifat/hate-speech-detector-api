import os
from dotenv import load_dotenv
import pandas as pd
from flask import Flask, request, jsonify
import functools
import datetime
from detoxify import Detoxify
import logging
import torch
from flask_caching import Cache
import pickle

load_dotenv()

ENABLE_API_TOKEN = os.getenv("ENABLE_API_TOKEN", "false") == "true"
API_TOKEN = os.getenv("API_TOKEN", "")
APP_ENV = os.getenv("APP_ENV", "production")
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = os.getenv("LISTEN_PORT", "7860")

CUSTOM_MODEL_PATH = os.getenv(
    "CUSTOM_MODEL_PATH",
    os.path.dirname(os.path.abspath(__file__))
    + "/experiments/model_voting_partial_best.pkl",
)
CUSTOM_VECTORIZER_PATH = os.getenv(
    "CUSTOM_VECTORIZER_PATH",
    os.path.dirname(os.path.abspath(__file__))
    + "/experiments/vectorizer_count_no_stop_words.pkl",
)

DETOXIFY_MODEL = os.getenv("DETOXIFY_MODEL", "unbiased-small")
HATE_SPEECH_MODEL = os.getenv("HATE_SPEECH_MODEL", "detoxify")
CACHE_DURATION_SECONDS = int(os.getenv("CACHE_DURATION_SECONDS", 60))
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "false") == "true"
POTENTIAL_TOXIC_WORDS = list(
    filter(None, os.getenv("POTENTIAL_TOXIC_WORDS", "").split(","))
)
HYBRID_THRESOLD_CHECK = float(os.getenv("HYBRID_THRESOLD_CHECK", 0.5))
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "auto")
APP_VERSION = "0.2.0"

# Setup logging configuration
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
if APP_ENV == "production":
    logging.basicConfig(
        level=logging.INFO,
        datefmt=LOGGING_DATE_FORMAT,
        format=LOGGING_FORMAT,
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        datefmt=LOGGING_DATE_FORMAT,
        format=LOGGING_FORMAT,
    )

if ENABLE_API_TOKEN and API_TOKEN == "":
    raise Exception("API_TOKEN is required if ENABLE_API_TOKEN is enabled")

if TORCH_DEVICE == "auto":
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch_device = TORCH_DEVICE

if HATE_SPEECH_MODEL in ["hybrid", "detoxify"]:
    model = Detoxify(DETOXIFY_MODEL, device=torch_device)

if HATE_SPEECH_MODEL in ["hybrid", "custom"]:
    try:
        with open(CUSTOM_VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        vectorizer = None

    try:
        with open(CUSTOM_MODEL_PATH, "rb") as f:
            model_custom = pickle.load(f)
    except Exception as e:
        raise e


app = Flask(__name__)

cache_config = {
    "DEBUG": True if APP_ENV != "production" else False,
    "CACHE_TYPE": "SimpleCache" if ENABLE_CACHE else "NullCache",
    "CACHE_DEFAULT_TIMEOUT": CACHE_DURATION_SECONDS,  # Cache duration in seconds
}
cache = Cache(config=cache_config)
cache.init_app(app)


def is_valid_api_key(api_key):
    if api_key == API_TOKEN:
        return True
    else:
        return False


def api_required(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if ENABLE_API_TOKEN:
            if request.json:
                api_key = request.json.get("api_key")
            else:
                return {"message": "Please provide an API key"}, 400
            # Check if API key is correct and valid
            if request.method == "POST" and is_valid_api_key(api_key):
                return func(*args, **kwargs)
            else:
                return {"message": "The provided API key is not valid"}, 403
        else:
            return func(*args, **kwargs)

    return decorator


def make_key_fn():
    """A function which is called to derive the key for a computed value.
       The key in this case is the concat value of all the json request
       parameters. Other strategy could to use any hashing function.
    :returns: unique string for which the value should be cached.
    """
    user_data = request.get_json()
    return ",".join([f"{key}={value}" for key, value in user_data.items()])


def perform_hate_speech_analysis(query):
    result = {}
    df = pd.DataFrame(model.predict(query), index=[0])
    columns = df.columns

    for i, label in enumerate(columns):
        result[label] = df[label][0].round(3).astype("float")

    return result


def perform_hate_speech_analysis_custom(query):
    query_vector = vectorizer.transform([query]) if vectorizer != None else ["query"]

    result = {
        "identity_attack": 0.0,
        "insult": 0.0,
        "obscene": 0.0,
        "severe_toxicity": 0.0,
        "sexual_explicit": 0.0,
        "threat": 0.0,
        "toxicity": 0.0,
    }

    temp_result = model_custom.predict_proba(query_vector)
    result["toxicity"] = temp_result[0][1].round(3).astype("float")

    return result


def perform_hate_speech_analysis_hybrid(
    query, thresold_check=0.5, potential_toxic_words=[]
):
    result = {
        "identity_attack": 0.0,
        "insult": 0.0,
        "obscene": 0.0,
        "severe_toxicity": 0.0,
        "sexual_explicit": 0.0,
        "threat": 0.0,
        "toxicity": 0.0,
    }
    temp_result_custom = perform_hate_speech_analysis_custom(query)

    has_potential_toxic_word = False
    for word in potential_toxic_words:
        if word in query:
            has_potential_toxic_word = True
            break

    if temp_result_custom["toxicity"] > thresold_check or has_potential_toxic_word:
        temp_result_detoxify = perform_hate_speech_analysis(query)
        if temp_result_detoxify["toxicity"] > thresold_check:
            result = temp_result_detoxify
    else:
        result = temp_result_custom

    return result


@app.errorhandler(Exception)
def handle_exception(error):
    res = {"error": str(error)}
    return jsonify(res)


@app.route("/predict", methods=["POST"])
@api_required
@cache.cached(make_cache_key=make_key_fn)
def predict():
    data = request.json
    q = data["q"]
    start_time = datetime.datetime.now()

    if HATE_SPEECH_MODEL == "custom":
        result = perform_hate_speech_analysis_custom(q)
    elif HATE_SPEECH_MODEL == "hybrid":
        result = perform_hate_speech_analysis_hybrid(
            q, HYBRID_THRESOLD_CHECK, POTENTIAL_TOXIC_WORDS
        )
    else:
        result = perform_hate_speech_analysis(q)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    logging.debug("elapsed predict time: %s", str(elapsed_time))
    return jsonify(result)


@app.route("/", methods=["GET"])
def index():
    response = {"message": "Use /predict route to get prediction result"}
    return jsonify(response)


@app.route("/app_version", methods=["GET"])
def app_version():
    response = {"message": "This app version is ".APP_VERSION}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host=LISTEN_HOST, port=LISTEN_PORT)
