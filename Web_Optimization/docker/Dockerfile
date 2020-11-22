# Builder
FROM python:3.8.5-buster as builder

WORKDIR /app

COPY Web_Optimization/docker/requirements.txt /app
RUN pip3 install -r requirements.txt

# Runner
FROM python:3.8.5-slim-buster as runner
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /usr/local/app

RUN apt update \
    && apt install -y libpq5 libxml2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./Web_Optimization /usr/local/app/Web_Optimization
COPY ./Web_Optimization/.streamlit /usr/local/app/.streamlit

RUN mkdir /home/appusr
RUN useradd -r -s /usr/local appusr
RUN chown -R appusr:appusr /home/appusr
USER appusr

EXPOSE 5000
CMD [ "streamlit", "run", "Web_Optimization/app.py" ]
