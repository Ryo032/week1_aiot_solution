FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir pandas numpy
COPY clean_stats.py .
CMD ["python", "clean_stats.py"]