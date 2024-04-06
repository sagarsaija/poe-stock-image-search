import argparse
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


def add_subtitle(video_file, caption):
    # Load the video file
    clip = VideoFileClip(video_file)

    # Create a text clip. You can customize the font, fontsize, color, etc.
    txt_clip = TextClip(caption, fontsize=70, color='white', font=" AvantGarde-Book")

    # Position the text in the center at the bottom of the screen
    txt_clip = txt_clip.set_position(
        ('center', 0.9), relative=True).set_duration(clip.duration)

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
