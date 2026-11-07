FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/opt/home \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

# Модели скачиваются при сборке и "запекаются" в образ (~10 МБ),
# чтобы контейнер стартовал без интернета к GitHub.
ARG DETECTOR_MODEL=yolo-v9-t-384-license-plate-end2end
ARG OCR_MODEL=cct-xs-v2-global-model
RUN mkdir -p $HOME && python -c "\
from open_image_models import create_detector; \
from fast_plate_ocr import LicensePlateRecognizer; \
create_detector('${DETECTOR_MODEL}'); LicensePlateRecognizer('${OCR_MODEL}', device='cpu')"

COPY app ./app
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
