from openai import OpenAI
from src.config import Config
from data_process.utils_dp import *
from src.prompts import *
from typing import Dict
import json

def model_init(config: Config, provider: str = "novita") -> OpenAI:
    if provider == "novita":
        llm = OpenAI(
            api_key=config.novita_api_key,
            base_url=config.novita_api_url
        )
    elif provider == "qwen":
        llm = OpenAI(
            api_key=config.qwen_vl_api_key,
            base_url=config.qwen_vl_api_url
        )

    return llm

def address_rewrite(llm: OpenAI, raw_address: str, config: Config):
    prompt = address_rewrite_pt()
    response = llm.chat.completions.create(
        model=config.address_rewrite_model_name,
        stream=False,
        max_tokens=200,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": raw_address}
        ]
    )
    return response.choices[0].message.content

def get_discription(llm: OpenAI, image_path: str, img_info, config: Config):
    prompt = photo_description_pt(timestamp=img_info['time'], location=img_info['address'])
    img = encode_image(image_path)

    response = llm.chat.completions.create(
        model=config.img_description_model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"}, 
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )

    return response.choices[0].message.content

def parse_img_description(llm_response: str) -> Dict:
    try:
        start_idx = llm_response.find('{')
        end_idx = llm_response.rfind('}') + 1
        json_str = llm_response[start_idx:end_idx]
        
        parsed_data = json.loads(json_str)
        
        assert "description" in parsed_data
        assert "tags" in parsed_data
        
        return parsed_data
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}")