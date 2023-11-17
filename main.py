import os

from fastapi_poe import make_app
from modal import Image, Secret, Stub, asgi_app

from vicuna_13b_v15 import Vicuna13BV15

image = Image.debian_slim().pip_install_from_requirements("requirements.txt")
stub = Stub("vicuna-13b-v1.5-app")


@stub.function(image=image, secret=Secret.from_name("vicuna-13b-v1.5-secret"))
@asgi_app()
def fastapi_app():
    bot = Vicuna13BV15(TOGETHER_API_KEY=os.environ["TOGETHER_API_KEY"])
    app = make_app(bot, access_key=os.environ["POE_ACCESS_KEY"])
    return app
