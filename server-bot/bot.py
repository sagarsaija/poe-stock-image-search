from __future__ import annotations

from typing import AsyncIterable

import os
import fal_client
import fastapi_poe as fp
import httpx
from dataclasses import dataclass
import time
import requests

STOCK_IMAGE_ACCESS_KEY = os.getenv("STOCK_IMAGE_POE_ACCESS_KEY")
FAL_KEY = os.getenv("FAL_KEY")
PREXEL_KEY = os.getenv("PREXEL_KEY")

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


app = fp.make_app([
    StockImage(
        path = "/stock-image",
        access_key = STOCK_IMAGE_ACCESS_KEY,
    ),
])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
