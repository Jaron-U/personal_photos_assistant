import os
from dotenv import load_dotenv
class Config:
    load_dotenv()
    novita_api_url = "https://api.novita.ai/v3/openai"
    novita_api_key = os.getenv("NOVITA_API_KEY")

    qwen_vl_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_vl_api_key = os.getenv("QWEN_VL_API_KEY")

    google_api_key = os.getenv("GOOGLE_API_KEY")

    address_rewrite_model_name = "qwen/qwen-2-7b-instruct"
    # img_description_model_name = "qwen2-vl-7b-instruct"
    img_description_model_name = "qwen/qwen-2-vl-72b-instruct"
