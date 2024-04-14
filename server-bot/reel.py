from __future__ import annotations
from openai import OpenAI
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip, concatenate_videoclips

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
import textwrap


# POE_INFERENCE_API_KEY = os.getenv("POE_INFERENCE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def wrap_text(text, max_width):
    """
    Wraps text to the specified width.
    This is a simple wrapper around textwrap.fill
    """
    return textwrap.fill(text, width=max_width)


def calculate_max_width(text, video_width, font_size, padding_percentage=10):
    """
    Calculate the maximum width of the text in characters.
    This is a heuristic that you might need to adjust based on your specific font and use case.

    Args:
    - text: the text to be wrapped.
    - video_width: the width of the video.
    - font_size: the font size of the text.
    - padding_percentage: the padding on each side of the text as a percentage of video width.

    Returns:
    - The maximum width of the text in characters.
    """
    # Calculate the padding in pixels
    total_padding = video_width * (padding_percentage / 100) * 2

    # Estimate the max width in characters, this is a heuristic and might need adjustment
    # Assuming each character roughly takes up font_size * 0.6 pixels in width
    max_char_width = (video_width - total_padding) / (font_size * 0.6)
    return int(max_char_width)


def add_subtitle(clip, caption, font_size=70, padding_percentage=10):
    # Calculate the maximum width in characters for the text
    max_width = calculate_max_width(
        caption, clip.size[0], font_size, padding_percentage)

    # Wrap the caption text
    wrapped_caption = wrap_text(caption, max_width=max_width)

    # Create a text clip with the wrapped text
    txt_clip = TextClip(wrapped_caption, fontsize=font_size, color='white', font="Arial-Bold",
                        align='center', method='caption', size=(clip.size[0]*0.9, None),
                        stroke_width=2, stroke_color='blue')

    # Position the text in the center at the bottom of the screen
    txt_clip = txt_clip.set_position(
        ('center', 0.85), relative=True).set_duration(clip.duration)

    # Overlay the text on the original video
    video = CompositeVideoClip([clip, txt_clip])

    return video


FAL_KEY = os.getenv("FAL_KEY")

SECONDS_PER_SCENE = 3

STYLES = [
    "cinematic",
    "realistic",
    "illustration",
    "cartoon",
    "cottage core"
]


def create_video_from_images(image_files, captions, output_file):
    clips = [ImageClip(file).set_duration(SECONDS_PER_SCENE)
             for file in image_files]

    clips_with_caption = [
        add_subtitle(clip, " ".join(caption)) for (clip, caption) in zip(clips, captions)
    ]

    # clips with subtitles
    video = concatenate_videoclips(clips_with_caption, method="compose")
    video.write_videofile(output_file, fps=24)


def create_script(prompt: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
    You're an expert AI video editor and you're tasked with generating a title and script for a short story to be shared on social media.
    
    Please generate a storyboard and script for the following prompt. Include narration and a scene description. Do not include character voices, just voiceover narration at this stage. Thank you.
    
    The story is just 40 seconds so it's not a full-length movie. Do not create more than 6 scenes.
        
    Story Inspiration:
    {prompt}

    """
    messages = [
        {"role": "system", "content": "You are a video editor screenwriting and then storyboarding a short story video on YouTube Shorts / TikTok.",
            "role": "user", "content": prompt}
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )
    answer = chat_completion.choices[0].message.content.strip()

    return answer


def summarize_prompt(long_prompt: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
    Given a the following user-inputted prompt I need a short 30-40 word descriptive prompt for a video generation system. Make sure it is evocative and descriptive, stay away from adjectives that won't translate well visually.

    Examples of desired output:
    Example 1:
    "'Harry Potter' is a fantasy series about a young wizard, Harry Potter, who battles the dark wizard Voldemort, with themes of friendship, bravery, and good versus evil."

    Example 2:
    "Three of the world's South Asian hackers build a video generation app at a hackathon and the AGI comes alive. It looks like a rainbow kraken. It eats San Francisco."

    User Input:
    {long_prompt}

    Generated Short Prompt:
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = ""
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )
    answer = chat_completion.choices[0].message.content.strip()

    return answer


class Reel(fp.PoeBot):

    def __post_init__(self) -> None:
        super().__post_init__()
        self.fal_client = fal_client.AsyncClient(key=FAL_KEY)
        self.http_client = httpx.AsyncClient()

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        message = request.query[-1]
        user_input = message.content.strip()

        # if len(prompt) > 200:
        yield fp.PartialResponse(text=f"Translating user_input to visual story guideline...\n")
        guideline = summarize_prompt(user_input)
        yield fp.PartialResponse(text=f"{guideline}\n")

        yield fp.PartialResponse(text=f"Creating script...\n")
        script = create_script(guideline)
        yield fp.PartialResponse(text=f"{script}\n")
        
        # yield fp.PartialResponse(text=f"Extracting scenes...\n")

        words = guideline.split()

        # create upto 4 words per scene
        scenes = [words[i:i+4] for i in range(0, len(words), 4)]

        if len(scenes) > 30:
            yield fp.PartialResponse(text=f"Content too long, truncating to 30 scenes\n")

        scenes = scenes[:30]
        style = random.choice(STYLES)
        consistence_words = []
        for scene in scenes:
            word = random.choice([x for x in scene if x.lower() not in [
                                 "is", "the", "a", "an", "are", "will", "shall"]])
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
                    "negative_prompt": "lowres, bad anatomy, hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, text, black and white",
                    "image_size": {
                        "height": 1920,
                        "width": 1080,
                    },
                    "num_inference_steps": 50,
                },
            )
            coros.append(coro)

        responses = await asyncio.gather(*coros)

        for i, response in enumerate(responses):
            image_url = response["images"][0]["url"]
            attachment_upload_response = await self.post_message_attachment(
                message_id=request.message_id,
                download_url=image_url,
                is_inline=True,
            )
            yield fp.PartialResponse(text=f"![scene][{attachment_upload_response.inline_ref}]\n")
            sub_resopnse = requests.get(image_url)
            if sub_resopnse.status_code != 200:
                continue
            image_data = sub_resopnse.content
            with open(f"scene_{i}.jpg", "wb") as file:
                file.write(image_data)
            print(f"Downloaded scene_{i}.jpg")
            time.sleep(0.3)

        timestamped_filename = f"output_{int(time.time())}.mp4"

        create_video_from_images(
            [f"scene_{i}.jpg" for i in range(len(scenes))], scenes, timestamped_filename)
        with open(timestamped_filename, "rb") as file:
            file_data = file.read()
        video_upload_response = await self.post_message_attachment(
            message_id=request.message_id,
            file_data=file_data,
            filename=timestamped_filename,
            # is_inline=True,
        )
        yield fp.PartialResponse(text=f"Video Created!\n\n")
        # yield fp.PartialResponse(text=f"![video][{video_upload_response.inline_ref or ""}]\n\n", is_replace_response=True)
