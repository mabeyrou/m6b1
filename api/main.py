from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from routes import router

app = FastAPI()

app.include_router(router)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

logger.remove()

logger.add("./logs/dev_backend.log",
          rotation="10 MB",
          retention="7 days",
          compression="zip",
          level="TRACE",
          enqueue=True,
          format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")