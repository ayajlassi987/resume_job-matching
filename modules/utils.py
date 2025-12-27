import os
import re


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_txt_folder(folder_path: str, limit: int = None):
    """
    Load up to `limit` .txt files from a folder.
    If limit is None, load all.
    """
    texts = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for i, filename in enumerate(os.listdir(folder_path)):
        if not filename.endswith(".txt"):
            continue

        if limit is not None and i >= limit:
            break

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())

    return texts
