import logging
import time

logger = logging.getLogger("uvicorn")

for _ in range(60):
    logger.info("Hello World!")
    print("Hello World!")
    time.sleep(1)
