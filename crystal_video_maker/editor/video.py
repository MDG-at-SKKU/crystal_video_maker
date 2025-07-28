# video.py
import moviepy.editor as mpy
import imageio
import io


def video_maker(image_bytes_list, name="video", format="mp4", fps=1):
    img_arrays = [imageio.imread(io.BytesIO(byte)) for byte in image_bytes_list]
    clip = mpy.ImageSequenceClip(img_arrays, fps=fps)
    clip.write_videofile(f"{name}.{format}", fps=fps)
