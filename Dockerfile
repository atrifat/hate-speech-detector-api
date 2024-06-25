FROM python:3.10-bookworm as builder

WORKDIR /builder

RUN addgroup --gid 1000 user
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user

ENV USER=user
ENV HOME=/home/user

RUN python3 -mvenv venv && ./venv/bin/pip install --no-cache-dir --upgrade pip

COPY requirements.txt requirements.txt

RUN ./venv/bin/pip install -U --no-cache-dir -r requirements.txt

FROM python:3.10-bookworm as runner

WORKDIR /app

RUN addgroup --gid 1000 user
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user

ENV USER=user
ENV HOME=/home/user

COPY --from=builder --chown=user:user /builder/venv /app/venv

COPY --chown=user:user app.py app.py

RUN mkdir -p /app/experiments
COPY --chown=user:user experiments/*.pkl experiments/ 

RUN chown -R user:user /app && chown -R user:user /home/user

USER user

ENV ENABLE_API_TOKEN=false
ENV API_TOKEN=
ENV APP_ENV=production
ENV LISTEN_HOST=0.0.0.0
ENV LISTEN_PORT=7860

EXPOSE $LISTEN_PORT

ENTRYPOINT [ "./venv/bin/python" , "app.py" ]
