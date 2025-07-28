# image.py
from crystal_video_maker.utils import check_name_available
from tqdm import tqdm

def prepare_image_name_with_checker(
    name="fig", number=0, numbering_rule="000", format="png"
):
    name_is_impossible = True
    img_number = number
    while name_is_impossible:
        img_name = f"{name}-{img_number:0{numbering_rule.count("0")}}.{format}"
        name_is_impossible = not check_name_available(img_name)
        if name_is_impossible:
            img_number += 1
        else:
            break
    return img_name


def prepare_iamge_name_with_sequence(
    length=1, name="fig", numbering_rule="000", format="png"
):
    name_list = [
        f"{name}-{i:0{numbering_rule.count("0")}}.{format}" for i in range(length)
    ]
    return name_list


def prepare_image_byte(figs) -> list:
    image_byte_list = []

    if type(figs) != list:
        figs = [figs]

    for fig in tqdm(figs):
        image_byte_list.append(fig.to_image(format="png"))
    return image_byte_list


def save_as_image_file(
    figs=None,
    name="fig",
    number=None,
    numbering_rule="000",
    format="png",
    with_check=False,
):
    if type(figs) != list:
        figs = [figs]

    if with_check:
        image_file_name_list = prepare_iamge_name_with_sequence(
            figs=figs, name=name, numbering_rule=numbering_rule, format=format
        )
        for i in tqdm(range(len(figs))):
            fig = figs[i]
            fig.write_image(f"result/{image_file_name_list[i]}.{format}")
    else:
        for i in tqdm(range(len(figs))):
            image_file_name = prepare_image_name_with_checker(
                name=name, number=number, numbering_rule=numbering_rule, format=format
            )
            fig = figs[i]
            fig.write_image(f"result/{image_file_name}.{format}")
