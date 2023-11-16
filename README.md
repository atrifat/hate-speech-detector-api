# hate-speech-detector-api

A Simple PoC (Proof of Concept) of Hate-speech (Toxic content) Detector API Server using model from [detoxify](https://github.com/unitaryai/detoxify). Detoxify (unbiased model) achieves score of 93.74% compared to top leaderboard score with 94.73% in [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification).

## Requirements

Python 3.9 or Python 3.10 is required to run the app. There is [bug/issue](https://github.com/unitaryai/detoxify/issues/94) for Python 3.11 or higher version affecting detoxify library.

## Getting Started

You can start by cloning this repository to run or modify it locally

```
git clone https://github.com/atrifat/hate-speech-detector-api
cd hate-speech-detector-api
```

Create virtual environment using venv, pyenv, or conda. This is an example using venv to create and activate the environment:

```
python3 -m venv venv
source venv/bin/activate
```

install its dependencies

```
pip install -U -r requirements.txt
```

and run it using command

```
python3 app.py
```

You can also copy `.env.example` to `.env` file and change the environment value based on your needs before running the app.

There is also Dockerfile available if you want to build docker image locally. If you don't want to build docker image locally, you can use the published version in [ghcr.io/atrifat/hate-speech-detector-api](https://github.com/atrifat/hate-speech-detector-api/pkgs/container/hate-speech-detector-api).

Run it:

```
docker run --init --env-file .env -p 7860:7860 -it ghcr.io/atrifat/hate-speech-detector-api
```

or run it in the background (daemon):

```
docker run --init --env-file .env -p 7860:7860 -it --name hate-speech-detector-api -d ghcr.io/atrifat/hate-speech-detector-api
```

If you want to test the API server, you can use GUI tools like [Postman](https://www.postman.com/) or using curl.

```
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"api_key":"your_own_api_key_if_you_set_them", "q":"hello world good morning"}' \
  http://localhost:7860/predict
```

The result of classification will be shown as follow (Example: using unbiased-small model):

```
{
    "identity_attack":0.0,
    "insult":0.0,
    "obscene":0.0,
    "severe_toxicity":0.0,
    "sexual_explicit":0.0,
    "threat":0.0,
    "toxicity":0.0010000000474974513
}
```

## License

MIT License

Copyright (c) 2023 Rif'at Ahdi Ramadhani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

Rif'at Ahdi Ramadhani (atrifat)
