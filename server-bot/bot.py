from __future__ import annotations

from typing import AsyncIterable

import os
import fal_client
import fastapi_poe as fp
import httpx
from dataclasses import dataclass
import time

POE_ACCESS_KEY = os.getenv("POE_ACCESS_KEY")
FAL_KEY = os.getenv("FAL_KEY")


class VideoMaker(fp.PoeBot):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.fal_client = fal_client.AsyncClient(key=FAL_KEY)
        self.http_client = httpx.AsyncClient()

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        message = request.query[-1]
        prompt = message.content

        yield fp.PartialResponse(text="Searching images\n")
        for i in range(3):
            yield fp.PartialResponse(text=".")
            # todo: search in pexel
            with open(f"static/sample{i+1}.jpeg", "rb") as f:
                file_data = f.read()
            attachment_upload_response = await self.post_message_attachment(
                message_id=request.message_id,
                file_data=file_data,
                filename=f"sample{i+1}.jpeg",
                is_inline=True,
            )
            yield fp.PartialResponse(
                text=f"![sample][{attachment_upload_response.inline_ref}]\n\n"
            )
            time.sleep(0.3)
        yield fp.PartialResponse(text="\n")
        
        yield fp.PartialResponse(text="Creating images\n")
        for i in range(3):
            yield fp.PartialResponse(text=".")
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


bot = VideoMaker()
app = fp.make_app(bot, POE_ACCESS_KEY)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
