import uuid


def uuid_str(digits=32) -> str:
    return uuid.uuid4().hex[:digits]