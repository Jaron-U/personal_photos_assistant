import requests
from FlagEmbedding import FlagModel
import transformers
transformers.logging.set_verbosity_error()

def create_search_query(query_params, embed_model: FlagModel, sort_params=None):
    must_conditions = []
    should_conditions = [] 

    # time search
    if 'time' in query_params:
        time_query = query_params['time']
        if isinstance(time_query, dict):
            if 'range' in time_query:
                must_conditions.append({
                    "range": {
                        "time": time_query['range']
                    }
                })
            # 上一次，最近一次等字段
            elif 'latest' in time_query and time_query['latest']:
                sort_params = sort_params or []
                sort_params.append({"time": {"order": "desc"}})
    
    # location search
    if 'location' in query_params:
        loc_query = {
            "bool": {
                "should": [
                    {"match": {"location": query_params['location']}},
                    {"term": {"location.keyword": query_params['location']}}
                ]
            }
        }
        if query_params.get('location_must', True):
            must_conditions.append(loc_query)
        else:
            should_conditions.append(loc_query)
    
    # discription search
    if 'description' in query_params:
        desc_query = {
            "bool": {
                "should": [
                    {"match": {"description": {
                        "query": query_params['description'],
                        "boost": 2.0
                    }}},
                    {"match": {"tags": {
                        "query": query_params['description'],
                        "boost": 1.0
                    }}}
                ]
            }
        }
        must_conditions.append(desc_query)

        descri_vec = embed_model.encode(query_params['description']).tolist()
        should_conditions.append({
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'description_vector') + 1.0",
                    "params": {"query_vector": descri_vec}
                }
            }
        })
    
    # tags search
    if 'tags' in query_params:
        tags_query = {"terms": {"tags": query_params['tags']}}
        if query_params.get('tags_must', True):
            must_conditions.append(tags_query)
        else:
            should_conditions.append(tags_query)
    
    # photo's type and emotion search
    for field in ['photo_type', 'emotion']:
        if field in query_params:
            term_query = {"term": {field: query_params[field]}}
            if query_params.get(f'{field}_must', True):
                must_conditions.append(term_query)
            else:
                should_conditions.append(term_query)
    
    query = {
        "query": {
            "bool": {
                "must": must_conditions,
                "should": should_conditions,
                "minimum_should_match": 1 if should_conditions else 0
            }
        }
    }

    # add sort
    if sort_params:
        query['sort'] = sort_params

    if 'size' in query_params:
        query['size'] = 5
    else:
        query['size'] = 200

    return query

def get_search_result(query_params, embed_model, top_k, 
                      db_url = "http://localhost:9200/photos_data/_search", sort_params=None):
    query = create_search_query(query_params, embed_model, sort_params)
    response = requests.post(db_url, json=query)
    result = response.json()
    hits = result['hits']['hits']
    context = {}
    contexts = [len(hits)]
    for hit in hits[:top_k]:
        source = hit['_source']
        context = {}
        context["description"] = source.get('description')
        context["file_path"] = source.get('file_path')
        context["time"] = source.get('time')
        context["location"] = source.get('location')
        context["tags"] = source.get('tags')
        context["photo_type"] = source.get('photo_type')
        context["emotion"] = source.get('emotion')
        contexts.append(context)

    return contexts

if __name__ == "__main__":
    embed_model = FlagModel(
        model_name_or_path = 'BAAI/bge-large-zh-v1.5',
        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        use_fp16=True
    )
    query_params = {
        "description": "滑雪",
        "tags": ["滑雪"],
        "size": 5
    }
    context = get_search_result(query_params, embed_model, top_k=10)
    print(context)
