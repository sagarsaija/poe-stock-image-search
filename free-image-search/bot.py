from __future__ import annotations

from typing import AsyncIterable

import os
import fal_client
import fastapi_poe as fp
import httpx
from dataclasses import dataclass
import requests

PEXEL_IMAGE_SEARCH_ACCESS_KEY = os.getenv("PEXEL_IMAGE_SEARCH_POE_ACCESS_KEY")
PREXEL_KEY = os.getenv("PREXEL_KEY")


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


bot = FreeImageSearcher()
app = fp.make_app(bot, PEXEL_IMAGE_SEARCH_ACCESS_KEY)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
