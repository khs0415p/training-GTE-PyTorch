import os
import shutil


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def remove_file(path: str):
    shutil.rmtree(path)

