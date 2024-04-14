from __future__ import annotations

from typing import AsyncIterable

import os
import fal_client
import fastapi_poe as fp
import httpx
from dataclasses import dataclass
import time
import requests

import reel

STOCK_IMAGE_ACCESS_KEY = os.getenv("STOCK_IMAGE_POE_ACCESS_KEY")
FAL_KEY = os.getenv("FAL_KEY")
PREXEL_KEY = os.getenv("PREXEL_KEY")
PEXEL_IMAGE_SEARCH_ACCESS_KEY = os.getenv("PEXEL_IMAGE_SEARCH_POE_ACCESS_KEY")
REEL_ACCESS_KEY = os.getenv("REEL_POE_ACCESS_KEY") # freshly added

COUNT_SEARCH = 3
COUNT_CREATE = 3

class StockImage(fp.PoeBot):


    def __post_init__(self) -> None:
        super().__post_init__()
        self.fal_client = fal_client.AsyncClient(key=FAL_KEY)
        self.http_client = httpx.AsyncClient()

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        message = request.query[-1]
        prompt = message.content

        url = f'https://api.pexels.com/v1/search?query={prompt}&per_page={COUNT_SEARCH}'
        headers = {
            'Authorization': PREXEL_KEY
        }

        response = requests.get(url, headers=headers)
        prexel_search_data = response.json()

        image_links = []
        for photo in prexel_search_data['photos']:
            image_links.append(photo['src']['large'])


        yield fp.PartialResponse(text="Searching images\n")
        for image_link in image_links:
            attachment_upload_response = await self.post_message_attachment(
                message_id=request.message_id,
                download_url=image_link,
                is_inline=True,
            )
            yield fp.PartialResponse(
                text=f"![sample][{attachment_upload_response.inline_ref}]\n\n"
            )
            time.sleep(0.3)
        
        yield fp.PartialResponse(text="Creating images\n")
        for i in range(3):
            response = await self.fal_client.run(
                "fal-ai/fast-sdxl",
                arguments={
                    "prompt": f"a realistic {prompt}, cinematic, ultra hd, high quality, video, cinematic, high quality",
                    "negative_prompt": "illustraiton, cartoon, blurry, text, not cinematic",
                    "image_size": {
                        "height": 512,
                        "width": 512,
                    },
                    "num_inference_steps": 30,
                },
            )
            image_url = response["images"][0]["url"]
            attachment_upload_response = await self.post_message_attachment(
                message_id=request.message_id,
                download_url=image_url,
                is_inline=True,
            )
            yield fp.PartialResponse(
                text=f"![image][{attachment_upload_response.inline_ref}]\n\n"
            )
        print("done!")

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(
            allow_attachments=True,
            introduction_message=(
                "Welcome to the video maker bot (powered by fal.ai). Please provide me a prompt to "
                "start with or an image so i can generate a video from it."
            ),
        )


def search_pexels_images(query, per_page=15, api_key=PREXEL_KEY):
    url = f'https://api.pexels.com/v1/search?query={query}&per_page={per_page}'
    headers = {
        'Authorization': api_key
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    image_links = []
    for photo in data['photos']:
        image_links.append(photo['src']['large'])

    return image_links


class FreeImageSearcher(fp.PoeBot):
    def __post_init__(self) -> None:
        super().__post_init__()
        # self.fal_client = fal_client.AsyncClient(key=FAL_KEY)
        self.http_client = httpx.AsyncClient()

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        message = request.query[-1]

        yield fp.PartialResponse(text=f"Searching pexels for {message.content}...")

        photo_links = search_pexels_images(message.content)

        yield fp.PartialResponse(text=f"Found {len(photo_links)} photos. Here are the first 5: {photo_links[:5]}")

    async def download_images(self, image_links):
        for i, link in enumerate(image_links):
            async with self.http_client.stream("GET", link) as response:
                with open(f"/tmp/image_{i}.jpg", "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        message = request.query[-1]

        yield fp.PartialResponse(text=f"Searching pexels for {message.content}...")

        photo_links = search_pexels_images(message.content)

        yield fp.PartialResponse(text=f"Found {len(photo_links)} photos. Downloading images...")

        for photo_link in photo_links:
            attachment_upload_response = await self.post_message_attachment(
                message_id=request.message_id,
                download_url=photo_link,
                is_inline=True,
            )
            yield fp.PartialResponse(
                text=f"![image][{attachment_upload_response.inline_ref}]\n\n"
            )

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(
            allow_attachments=True,
            introduction_message=(
                "Welcome to the video maker bot (powered by fal.ai). Please provide me a prompt to "
                "start with or an image so i can generate a video from it."
            ),
        )



app = fp.make_app([
    StockImage(
        path = "/stock-image",
        access_key = STOCK_IMAGE_ACCESS_KEY,
    ),
    FreeImageSearcher(
        path = "/free-image-search",
        access_key=PEXEL_IMAGE_SEARCH_ACCESS_KEY,
    ),
    reel.Reel(
        path="/reel",
        access_key=REEL_ACCESS_KEY,
    ),
])

FLY_TASKS_APP = "poe-stock-image-search"
FLY_API_TOKEN = os.getenv("FLY_API_TOKEN")

print(f"FLY_API_TOKEN: {FLY_API_TOKEN[:3]}...{FLY_API_TOKEN[-3:]}")

headers = {
    "Authorization": f"Bearer {FLY_API_TOKEN}",
    "Content-Type": "application/json"
}

WORKER_IMAGE = "registry.fly.io/poe-stock-image-search:latest"

MACHINE_CONFIG = {
    "config": {
        "image": WORKER_IMAGE,
        "env": {
        },
        "processes": [{
            "name": "worker",
            "entrypoint": ["python"],
            "cmd": ["app/worker.py"]
        }]
    }
}

@app.get("/test")
async def test():
    print(f"FLY_API_TOKEN: {FLY_API_TOKEN[:3]}...{FLY_API_TOKEN[-3:]}")
    machine_config = dict(**MACHINE_CONFIG)
    machine_config["name"] = "test-image"
    print(f"requesting to create a machine")
    response = requests.post(f"https://api.machines.dev/v1/apps/{FLY_TASKS_APP}/machines", headers=headers, json=machine_config)
    response.raise_for_status()
    # store the machine id so we can use it later to check if the job has completed
    response = response.json()
    print(f"response: {response}")
    machine_id = response["id"]
    return {
        "machine_id": machine_id,
        # "task_id": task_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
