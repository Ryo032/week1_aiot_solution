import asyncio
import contextvars
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ==============================================================================
# 1. STRUCTURED JSON LOGGING & TRUY VẾT XUYÊN DỊCH VỤ (CONTEXTVAR)
# ==============================================================================
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="system")


class JsonLogFormatter(logging.Formatter):
    """Định dạng mỗi bản ghi log thành đúng 1 dòng JSON chuẩn ELK / Grafana Loki."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "service": "aiot-inference-service",
            "request_id": request_id_ctx.get(),
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_payload.update(record.extra_data)
        return json.dumps(log_payload, ensure_ascii=False)


logger = logging.getLogger("inference_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JsonLogFormatter())
logger.handlers = [handler]


# ==============================================================================
# 2. CƠ SỞ DỮ LIỆU LƯU VẾT LỊCH SỬ SUY LUẬN (SQLALCHEMY)
# ==============================================================================
engine = create_engine("sqlite:///./inference_history.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base: Any = declarative_base()


class InferenceLogDB(Base):
    """Bảng lưu vết lịch sử suy luận để kiểm toán và tái huấn luyện sau này."""

    __tablename__ = "inference_logs"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    model_version = Column(String)
    temperature = Column(Float)
    vibration = Column(Float)
    prediction = Column(String)
    latency_ms = Column(Float)


Base.metadata.create_all(bind=engine)


def save_inference_to_db(
    req_id: str,
    version: str,
    temp: float,
    vib: float,
    pred: str,
    latency_ms: float,
) -> None:
    """Hàm ghi DB chạy ở hậu trường (BackgroundTasks) để không chặn HTTP Response."""
    db = SessionLocal()
    try:
        record = InferenceLogDB(
            request_id=req_id,
            model_version=version,
            temperature=temp,
            vibration=vib,
            prediction=pred,
            latency_ms=latency_ms,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


# ==============================================================================
# 3. LỚP CACHE-ASIDE (REDIS KÈM FALLBACK RAM KHI TEST CỤC BỘ)
# ==============================================================================
class CacheClient:
    """Bọc Redis client; nếu chưa bật Redis Server sẽ tự động dùng dict RAM có TTL."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis: Any = None
        self._memory_cache: dict[str, tuple[float, str]] = {}
        try:
            import redis  # type: ignore[import-not-found]

            client = redis.from_url(redis_url, decode_responses=True, socket_timeout=0.5)
            client.ping()
            self._redis = client
            logger.info("Da ket noi thanh cong toi Redis Server.")
        except Exception:
            logger.warning("Khong tim thay Redis, chuyen sang In-Memory Cache du phong.")

    def make_key(self, model_version: str, features: dict[str, float]) -> str:
        """Tạo Cache Key bằng SHA-256 có gắn model_version chống Cache Pollution."""
        canonical_json = json.dumps(features, sort_keys=True, separators=(",", ":"))
        raw_str = f"{model_version}:{canonical_json}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if self._redis is not None:
            val = self._redis.get(key)
            return dict(json.loads(val)) if val else None
        item = self._memory_cache.get(key)
        if item and item[0] > time.time():
            return dict(json.loads(item[1]))
        return None

    def set_with_ttl(self, key: str, value: dict[str, Any], ttl_seconds: int = 60) -> None:
        serialized = json.dumps(value)
        if self._redis is not None:
            self._redis.setex(key, ttl_seconds, serialized)
        else:
            self._memory_cache[key] = (time.time() + ttl_seconds, serialized)


# ==============================================================================
# 4. MODEL REGISTRY ĐA PHIÊN BẢN & DYNAMIC BATCHING WORKER
# ==============================================================================
class MockAIModel:
    """Mô phỏng mô hình ML xử lý theo ma trận lô (B, 2) -> (B,)."""

    def __init__(self, version: str, temp_weight: float, vib_weight: float) -> None:
        self.version = version
        self.weights = np.array([temp_weight, vib_weight], dtype=np.float32)

    def predict_batch(self, batch_matrix: np.ndarray) -> list[dict[str, Any]]:
        time.sleep(0.015 + 0.002 * len(batch_matrix))
        scores = batch_matrix @ self.weights
        results: list[dict[str, Any]] = []
        for score in scores:
            label = "CRITICAL_FAULT" if float(score) > 50.0 else "NORMAL"
            results.append({"label": label, "risk_score": round(float(score), 3)})
        return results


class DynamicBatcher:
    """Gom các request lẻ thành ma trận lô (B, D) theo ngưỡng B_max hoặc T_wait."""

    def __init__(
        self,
        model: MockAIModel,
        max_batch_size: int = 8,
        max_wait_ms: int = 20,
        max_queue_size: int = 32,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_sec = max_wait_ms / 1000.0
        self.queue: asyncio.Queue[tuple[list[float], asyncio.Future[dict[str, Any]]]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._worker_task = asyncio.create_task(self._batch_worker_loop())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()

    async def submit(self, feature_vec: list[float]) -> dict[str, Any]:
        """Đẩy 1 vector đầu vào kèm phiếu hẹn Future vào hàng đợi."""
        if self.queue.full():
            raise HTTPException(
                status_code=429,
                detail="Inference queue is full. Please retry later.",
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        await self.queue.put((feature_vec, future))
        return await future

    async def _batch_worker_loop(self) -> None:
        """Vòng lặp chạy nền gom lô theo điều kiện OR (đủ B_max HOẶC hết T_wait)."""
        while True:
            first_item = await self.queue.get()
            batch_items = [first_item]
            deadline = time.perf_counter() + self.max_wait_sec

            while len(batch_items) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch_items.append(next_item)
                except asyncio.TimeoutError:
                    break

            vectors = np.array([item[0] for item in batch_items], dtype=np.float32)
            futures = [item[1] for item in batch_items]

            try:
                predictions = await asyncio.to_thread(self.model.predict_batch, vectors)
                for fut, pred in zip(futures, predictions):
                    if not fut.done():
                        fut.set_result(pred)
            except Exception as exc:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)


# ==============================================================================
# 5. KHỞI TẠO FASTAPI, VALIDATION, BACKPRESSURE & GLOBAL EXCEPTION HANDLER
# ==============================================================================
MODEL_REGISTRY: dict[str, DynamicBatcher] = {}
CACHE = CacheClient()
CONCURRENCY_SEMAPHORE = asyncio.Semaphore(16)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Nạp sẵn toàn bộ các phiên bản mô hình vào RAM trước khi mở cổng nhận traffic."""
    MODEL_REGISTRY["v1.0.0"] = DynamicBatcher(MockAIModel("v1.0.0", 1.0, 2.0))
    MODEL_REGISTRY["v2.0.0"] = DynamicBatcher(MockAIModel("v2.0.0", 1.2, 2.5))
    for batcher in MODEL_REGISTRY.values():
        batcher.start()
    yield
    for batcher in MODEL_REGISTRY.values():
        await batcher.stop()


app = FastAPI(title="AIoT Production Inference Service - Week 6", lifespan=lifespan)


class SensorPayload(BaseModel):
    """Kiểm tra chặt chẽ kiểu dữ liệu và biên vật lý ngay ở cổng vào."""

    temperature: float = Field(..., ge=-50.0, le=200.0, description="Nhiệt độ (°C)")
    vibration: float = Field(..., ge=0.0, le=100.0, description="Độ rung (mm/s)")

    @field_validator("temperature", "vibration")
    @classmethod
    def reject_nan_or_inf(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Gia tri cam bien khong duoc la NaN hoac Infinity")
        return v


@app.middleware("http")
async def tracing_middleware(request: Request, call_next: Any) -> Any:
    """Đọc X-Request-ID từ Go Gateway (hoặc tự tạo mới) và gắn vào ContextVar."""
    req_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:8]}")
    token = request_id_ctx.set(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        request_id_ctx.reset(token)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Bắt mọi lỗi bất ngờ: ghi log nội bộ nhưng ẩn Stack Trace khỏi HTTP Response."""
    logger.error("Unhandled exception: %s", str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "request_id": request_id_ctx.get()},
    )


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Endpoint kiểm tra sức khỏe và danh sách phiên bản mô hình đang phục vụ."""
    return {"status": "ok", "loaded_versions": list(MODEL_REGISTRY.keys())}


@app.post("/v1/predict")
async def predict(
    payload: SensorPayload,
    background_tasks: BackgroundTasks,
    x_model_version: str = Header(default="v2.0.0", alias="X-Model-Version"),
) -> dict[str, Any]:
    start_time = time.perf_counter()

    if x_model_version not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{x_model_version}' not found.",
        )

    features = {"temperature": payload.temperature, "vibration": payload.vibration}
    cache_key = CACHE.make_key(x_model_version, features)

    cached_result = CACHE.get(cache_key)
    if cached_result is not None:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Prediction served from cache",
            extra={
                "extra_data": {
                    "model_version": x_model_version,
                    "cache_hit": True,
                    "latency_ms": latency_ms,
                }
            },
        )
        return {
            **cached_result,
            "model_version": x_model_version,
            "cache_hit": True,
            "latency_ms": latency_ms,
        }

    async with CONCURRENCY_SEMAPHORE:
        batcher = MODEL_REGISTRY[x_model_version]
        result = await batcher.submit([payload.temperature, payload.vibration])

    CACHE.set_with_ttl(cache_key, result, ttl_seconds=60)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    background_tasks.add_task(
        save_inference_to_db,
        request_id_ctx.get(),
        x_model_version,
        payload.temperature,
        payload.vibration,
        result["label"],
        latency_ms,
    )

    logger.info(
        "Prediction computed by model",
        extra={
            "extra_data": {
                "model_version": x_model_version,
                "cache_hit": False,
                "latency_ms": latency_ms,
            }
        },
    )
    return {
        **result,
        "model_version": x_model_version,
        "cache_hit": False,
        "latency_ms": latency_ms,
    }
