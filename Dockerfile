# --- HỘP 1: BUILDER (Chuyên cài thư viện) ---
FROM python:3.10-slim AS builder
ENV PYTHONUNBUFFERED=1
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- HỘP 2: RUNNER (Hộp sạch đem đi chạy thực tế) ---
FROM python:3.10-slim AS runner
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Tạo tài khoản thường appuser (UID 10001) thay cho root
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /sbin/nologin appuser && \
    chown -R appuser:appgroup /app

# Bốc thư viện từ hộp builder sang và copy code kèm quyền sở hữu
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup . /app

USER appuser

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os; assert os.path.exists('/app/clean_stats.py')" || exit 1

CMD ["python", "clean_stats.py"]