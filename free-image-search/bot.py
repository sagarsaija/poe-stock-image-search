from __future__ import annotations

from typing import AsyncIterable

import os
import fal_client
import fastapi_poe as fp
import httpx
from dataclasses import dataclass
import requests

bot = FreeImageSearcher()
app = fp.make_app(bot, PEXEL_IMAGE_SEARCH_ACCESS_KEY)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
