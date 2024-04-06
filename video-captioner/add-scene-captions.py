import argparse
import textwrap
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


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

def add_subtitle(video_file, caption, font_size=70, padding_percentage=10):
    # Load the video file
    clip = VideoFileClip(video_file)

    # Calculate the maximum width in characters for the text
    max_width = calculate_max_width(caption, clip.size[0], font_size, padding_percentage)

    # Wrap the caption text
    wrapped_caption = wrap_text(caption, max_width=max_width)

    # Create a text clip with the wrapped text
    txt_clip = TextClip(wrapped_caption, fontsize=font_size, color='white', font="Arial-Bold",
                        align='center', method='caption', size=(clip.size[0]*0.9,None))

    # Position the text in the center at the bottom of the screen
    txt_clip = txt_clip.set_position(('center', 0.85), relative=True).set_duration(clip.duration)

    # Overlay the text on the original video
    video = CompositeVideoClip([clip, txt_clip])

    # Output file name
    output_file = f"subtitled_{video_file}"

    # Write the result to a file
    video.write_videofile(output_file, codec='libx264', audio_codec='aac')

    print(f"Subtitled video saved as {output_file}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Add subtitles to a video.")
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("caption", help="Caption text to add as subtitle")

    # Parse arguments
    args = parser.parse_args()

    # Add subtitle to video
    add_subtitle(args.video_file, args.caption)
