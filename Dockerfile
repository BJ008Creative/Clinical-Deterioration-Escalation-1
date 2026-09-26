# *** UNTESTED *** No `docker` CLI is available in the build sandbox this
# was written in, so this has never actually been built or run. Verify with
# `docker build -t vigil .` in a normal environment before relying on it.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] streamlit

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000 8501

# Default: the API. Override the command to run the dashboard instead:
#   docker run <image> streamlit run src/vigil/ui/dashboard.py --server.address 0.0.0.0
CMD ["uvicorn", "vigil.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
