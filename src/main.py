from src.config import Config
from src.llmodel import LLModel
from src.prompts import *
from src.retrieve import get_search_result
from FlagEmbedding import FlagModel
import base64, json

def llmodel_init(config: Config):
    query_anlysis_model = LLModel(
        base_url=config.novita_api_url,
        api_key=config.novita_api_key,
        model_name=config.img_description_model_name,
        sys_prompt=multimodal_prompt,
        stream=False,
        max_tokens=5000,
        max_round_dialog=1,
        min_round_dialog=1
    )

    generate_model = LLModel(
        base_url=config.novita_api_url,
        api_key=config.novita_api_key,
        model_name=config.address_rewrite_model_name,
        sys_prompt=generate_pt,
        stream=True,
        max_tokens=2500,
        max_round_dialog=1,
        min_round_dialog=1
    )

    return query_anlysis_model, generate_model

def analyze_query(query: str, query_anlysis_model: LLModel, photo=None):
    query_anlysis_model.add_user_message(query, photo)
    response = query_anlysis_model.get_response(print_response=False)
    query_anlysis_model.add_assistant_message(response)
    print(response)
    # query_anlysis_model.print_messages()
    
    ## extract the query params from the response
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        json_str = response[start_idx:end_idx]
        search_para = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return search_para

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate(query: str, generate_model: LLModel, retrieved_context: dict, photo_description=None):
    photos_num = retrieved_context[0]
    if not photo_description:
        combined_context = f"user query: {query}\n retrieved context: totally get {photos_num} photos. These are details: {retrieved_context}"
    else:
        combined_context = f"user query: {query}, input photo discription: {photo_description}\n retrieved context: totally get {photos_num} photos. These are details: {retrieved_context}\n"
    generate_model.temp_add_user_message(combined_context)
    response = generate_model.get_response(print_response=True)
    generate_model.remove_last_user_message()
    generate_model.add_user_message(query)
    generate_model.add_assistant_message(response)
    return response

def run_bash(query_anlysis_model, generate_model, embedding_model):
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        img_path = input("input the image path or type 'no': ")
        if img_path.lower() != "no":
            try:
                with open(img_path, "rb") as f:
                    img = f.read()
                photo = encode_image(img_path)
            except FileNotFoundError:
                print("File not found")
                continue
        else:
            photo = None
        
        query_params = analyze_query(user_input, query_anlysis_model, photo)
        if not query_params:
            print("对不起，我无法理解您的查询。")
            continue
        if photo:
            photo_descri = query_params['image_description']
        search_params = query_params['search_params']
        retrieved_context = get_search_result(search_params, embedding_model, config.top_k)
        _ = generate(user_input, generate_model, retrieved_context, photo_descri)

if __name__ == "__main__":
    config = Config()
    query_anlysis_model, generate_model = llmodel_init(config)
    embed_model = FlagModel(
        model_name_or_path = 'BAAI/bge-large-zh-v1.5',
        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        use_fp16=True
    )
    run_bash(query_anlysis_model, generate_model, embed_model)
