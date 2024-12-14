from random import randint

from fastapi import UploadFile


async def image_to_number(image: UploadFile) -> int:
    rn = randint(0, 9)
    return rn
