import os

from fastapi_poe import make_app
from modal import Image, Secret, Stub, asgi_app

from nous_hermes_llama2_13b import NousHermesLlama213B

image = Image.debian_slim().pip_install_from_requirements("requirements.txt")
stub = Stub("nous-hermes-llama2-13b-hf-app")


@stub.function(image=image, secret=Secret.from_name("nous-hermes-l2-13b-secret"))
@asgi_app()
def fastapi_app():
    bot = NousHermesLlama213B(
        endpoint_url="https://yywfgbg0c2v2cczg.us-east-1.aws.endpoints.huggingface.cloud",
        model_name="NousResearch/Nous-Hermes-Llama2-13b",
        token=os.environ["HF_TOKEN"],
    )
    app = make_app(bot, api_key=os.environ["POE_API_KEY"])
    return app
