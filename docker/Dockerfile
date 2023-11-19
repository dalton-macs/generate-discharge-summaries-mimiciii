# Generate Brief Hospital Course Section of Patient Discharge Summaries.
# Author: Dalton Macres (dgmacres@wpi.edu)
# Date: 2023-11-19

FROM python:3.11.5-slim

WORKDIR /app/

LABEL name="gendis" \
    version="1.0.0" \
    maintainer="Dalton Macres (dgmacres@wpi.edu)"

ENV PIP_DEFAULT_TIMEOUT=100 \
    # allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
COPY dropdowns dropdowns
COPY recursive_summary.py .
COPY app.py .

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

# set streamlit config via env vars
ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_LOGGER_LEVEL="info"
ENV STREAMLIT_CLIENT_TOOLBAR_MODE="viewer"
ENV STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_THEME_BASE="light"
ENV STREAMLIT_THEME_PRIMARY_COLOR="#3383f6"

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]