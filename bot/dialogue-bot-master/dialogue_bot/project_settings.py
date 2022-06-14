import os

# import settings


def ensure_dir(path, is_filepath=None):
    """Ensures that the (parent-)directories of the given path exist"""

    if is_filepath is None:
        # use heuristic to determine this value
        dir_path, name = os.path.split(path)
        is_filepath = "." in name[1:-1]

    parent_dir = os.path.dirname(path) if is_filepath else path
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    return path


PROJECT_DIRPATH = os.getenv(
    "DBOT_PROJECT_DIRPATH", os.path.dirname(os.path.abspath(__file__))
)
DATA_DIRPATH = os.getenv("DBOT_DATA_DIRPATH", os.path.join(PROJECT_DIRPATH, "data"))

STATIC_DATA_DIRPATH = os.getenv(
    "DBOT_STATIC_DATA_DIRPATH", os.path.join(DATA_DIRPATH, "static")
)
GENERATED_DATA_DIRPATH = os.getenv(
    "DBOT_GENERATED_DATA_DIRPATH", os.path.join(DATA_DIRPATH, "generated")
)
