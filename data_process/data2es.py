from elasticsearch import Elasticsearch, helpers
import json, uuid
from FlagEmbedding import FlagModel
es = Elasticsearch('http://localhost:9200')
from datetime import datetime, timezone
from tqdm import tqdm

def create_index(index_name="photos_data"):
    create_index_body = {
        "settings": {
            "index": {
                "number_of_shards": 3,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "photo_id": {
                    "type": "keyword"
                },
                "file_path": {
                    "type": "keyword"
                },
                "time": {
                    "type": "date"
                },
                "location": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                "description": {
                    "type": "text"
                },
                "description_vector": {
                    "type": "dense_vector",
                    "dims": 1024
                },
                "tags": {
                    "type": "keyword"
                },
                "photo_type": {
                    "type": "keyword"
                },
                "emotion": {
                    "type": "keyword"
                },
                "create_time": {
                    "type": "date",
                    "format": "yyyy-MM-dd HH:mm:ss"
                },
            }
        }
    }

    # if the db is exit just delete it
    if es.indices.exists(index=index_name):
        print(f"delete index {index_name}")
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=create_index_body)
    print(f"create index {index_name}")

def embedding_description(description, model: FlagModel):
    return model.encode(description)

def create_es_action(index_name, record, description_vec):
    action = {
        "_index": index_name,
        "_id": str(uuid.uuid4()),
        "_source": {
            "photo_id": record['id'],
            "file_path": record['file_path'],
            "time": record['time'],
            "location": record['location'],
            "description": record['description'],
            "description_vector": description_vec,
            "tags": record['tags'],
            "photo_type": record['photo_type'],
            "emotion": record['emotion'],
            "create_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    return action

def data2es(index_name, embed_model, dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions = []
    for record in tqdm(data):
        description_vec = embedding_description(record['description'], embed_model)
        action = create_es_action(index_name, record, description_vec)
        actions.append(action)

    helpers.bulk(es, actions)
    print(f"index {index_name} done")

def main():
    model = FlagModel('BAAI/bge-large-zh-v1.5', 
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True)
    index_name = "photos_data"
    dataset_path = "photos_info.json"
    create_index(index_name)
    data2es(index_name, model, dataset_path)

if __name__ == "__main__":
    main()



