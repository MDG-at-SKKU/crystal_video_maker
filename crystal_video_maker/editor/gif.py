# gif.py
from PIL import Image
import io


def gif_maker(image_bytes_list):
    img_arrays = [Image.open(io.BytesIO(byte)) for byte in image_bytes_list]

    # 첫 번째 이미지를 기준으로 GIF 저장 (append_images로 나머지 추가)
    img_arrays[0].save(
        "animation.gif",
        save_all=True,
        append_images=img_arrays[1:],
        optimize=True,  # 파일 크기 최적화
        duration=100,  # 각 프레임 표시 시간 (ms, 0.5초)
        loop=0,  # 무한 반복 (0: 무한, 1: 한 번 등)
    )
