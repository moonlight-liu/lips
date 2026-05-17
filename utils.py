import os


def get_list(path) -> list:
    r"""Recursively read all files in root path"""
    image_list = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in files:
            if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]:
                image_list.append(os.path.join(root, f))
    image_list.sort()
    return image_list
