import os

from fastapi_poe import make_app
from modal import Image, Secret, Stub, asgi_app

from nous_hermes_llama2_13b import NousHermesLlama213B

image = Image.debian_slim().pip_install_from_requirements("requirements.txt")
stub = Stub("nous-hermes-llama2-13b-app")


@stub.function(image=image, secret=Secret.from_name("nous-hermes-l2-13b-secret"))
@asgi_app()
def fastapi_app():
    bot = NousHermesLlama213B(TOGETHER_API_KEY=os.environ["TOGETHER_API_KEY"])
    app = make_app(bot, api_key=os.environ["POE_API_KEY"])
    return app
