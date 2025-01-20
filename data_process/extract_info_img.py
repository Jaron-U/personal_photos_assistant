from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
from data_process.utils import *
from src.config import Config
from data_process.llm_generate import *
from tqdm import tqdm
import uuid

def get_img_info(image_path) -> dict:
    info = {}

    info['time'] = get_photo_datetime_exif(image_path)

    image = Image.open(image_path)
    exif_data = {}
    if image._getexif():
        for tag, value in image._getexif().items():
            tag_name = TAGS.get(tag, tag)
            exif_data[tag_name] = value
    
        gps_info = {}
        if 'GPSInfo' in exif_data:
            for key in exif_data['GPSInfo'].keys():
                decoded_key = GPSTAGS.get(key, key)
                gps_info[decoded_key] = exif_data['GPSInfo'][key]
        location = convert_to_degrees(gps_info)
        if location:
            latitude, longitude = location
            info['location'] = (latitude, longitude)
            return info
    
    info['location'] = None
    return info

def get_detail_info(image_path: str, google_api_key: str, address_rewrite_model: OpenAI, config: Config) -> dict:
    img_info = get_img_info(image_path)
    if img_info['location']:
        latitude, longitude = img_info['location']
        address_raw = gps_to_location_detailed(latitude, longitude, google_api_key)
        img_info['address'] = address_rewrite(address_rewrite_model, address_raw["formatted_address"], config)
    else:
        img_info['address'] = None
    return img_info

def save_img_info_json(data_path: str, output_path: str, config: Config):
    address_rewrite_model = model_init(config)
    img_description_model = model_init(config)

    files = os.listdir(data_path)
    json_data = []
    for file in tqdm(files):
        if file.endswith(".jpeg"):
            image_path = os.path.join(data_path, file)
            img_info = get_detail_info(image_path, config.google_api_key, address_rewrite_model, config)
            response = get_discription(img_description_model, image_path, img_info, config)
            parsed_data = parse_img_description(response)

            img_id = str(uuid.uuid4())
            json_data.append({
                "id": img_id,
                "file_path": image_path,
                "time": img_info['time'],
                "location": img_info['address'] if img_info['address'] else "未知",
                "description": parsed_data['description'],
                "tags": parsed_data['tags'],
                "photo_type": parsed_data['photo_type'],
                "emotion": parsed_data['emotion']
            })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    config = Config()
    data_path = "dataset"
    output_path = "photos_info.json"
    # save_img_info_json(data_path, output_path, config)
    print("Done!")


    # # image_path = "dataset/0B46E58A-99DD-4F49-95E7-3DC315A4A27E_1_105_c.jpeg"
    # image_path = "dataset/D3F1FAE3-79FD-4DD0-A489-F1712DDDF6C2_1_105_c.jpeg"
    # google_api_key=config.google_api_key
    # address_rewrite_model = model_init(config)
    # img_info = get_detail_info(image_path, google_api_key, address_rewrite_model, config)
    # print(img_info)
    # img_description_model = model_init(config, "qwen")
    # response = get_discription(img_description_model, image_path, img_info, config)
    # print(response)
    # parsed_data = parse_img_description(response)
    # print(parsed_data)
    