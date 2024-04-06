from __future__ import annotations

from typing import AsyncIterable

import os
import fal_client
import fastapi_poe as fp
import httpx
from dataclasses import dataclass
import time
import requests
import asyncio
import random

import moviepy.editor as mpy

FAL_KEY = os.getenv("FAL_KEY")

SECONDS_PER_SCENE = 3

STYLES = [
    "cinematic",
    "realistic",
    "illustration",
    "cartoon",
    "cottage core"
]

def create_video_from_images(image_files, output_file):
    clips = [mpy.ImageClip(file).set_duration(SECONDS_PER_SCENE) for file in image_files]
    video = mpy.concatenate_videoclips(clips, method="compose")
    video.write_videofile(output_file, fps=24)


class Reel(fp.PoeBot):

    def __post_init__(self) -> None:
        super().__post_init__()
        self.fal_client = fal_client.AsyncClient(key=FAL_KEY)
        self.http_client = httpx.AsyncClient()

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        message = request.query[-1]
        prompt = message.content

        words = prompt.split()

        # create upto 4 words per scene
        scenes = [words[i:i+4] for i in range(0, len(words), 4)]

        if len(scenes) > 30:
            yield fp.PartialResponse(text=f"Content too long, truncating to 30 scenes\n")

        scenes = scenes[:30]
        style = random.choice(STYLES)
        consistence_words = []
        for scene in scenes:
            word = random.choice([x for x in scene if x.lower() not in ["is", "the", "a", "an", "are", "will", "shall"]])
            consistence_words.append(word)
        consistence_words = " ".join(consistence_words)


        coros = []
        for scene in scenes:
            scene_description = " ".join(scene)
            scene_prompt = f"{scene_description} in {style} style, describing {consistence_words}"
            print(f"Asking for scene: {scene_prompt}")
            coro = self.fal_client.run(
                "fal-ai/fast-sdxl",
                arguments={
                    "prompt": scene_prompt,
                    "negative_prompt": "blurry, text",
                    "image_size": {
                        "height": 1920,
                        "width": 1080,
                    },
                    "num_inference_steps": 30,
                },
            )
            coros.append(coro)

        responses = await asyncio.gather(*coros)

        for i,response in enumerate(responses):
            image_url = response["images"][0]["url"]
            attachment_upload_response = await self.post_message_attachment(
                message_id=request.message_id,
                download_url=image_url,
            )
            yield fp.PartialResponse(text=f".")
            sub_resopnse = requests.get(image_url)
            if sub_resopnse.status_code != 200:
                continue
            image_data = sub_resopnse.content
            with open(f"scene_{i}.jpg", "wb") as file:
                file.write(image_data)
            print(f"Downloaded scene_{i}.jpg")

        create_video_from_images([f"scene_{i}.jpg" for i in range(len(scenes))], "output.mp4")
        with open("output.mp4", "rb") as file:
            file_data = file.read()
        video_upload_response = await self.post_message_attachment(
            message_id=request.message_id,
            file_data=file_data,
            filename="output.mp4",
            # is_inline=True,
        )
        yield fp.PartialResponse(text=f"![video][{video_upload_response.inline_ref}]\n\n", is_replace_response=True)

