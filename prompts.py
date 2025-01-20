
def photo_description_pt(timestamp=None, location=None):
    prompt = f"""
    已知信息：
    {f'拍摄时间：{timestamp}' if timestamp else '拍摄时间未知'}
    {f'拍摄地点：{location}' if location else '拍摄地点未知'}

    请从以下角度描述照片：
    1. 照片类型：[判断是自拍/风景/美食/合照/截图等]
    2. 画面内容：
       - 主要内容是什么
       - 有哪些人物（如果有），他们在做什么
       - 场景的关键特征（环境、光线、构图等）
    3. 情感和记忆：
       - 这个场景/时刻传达的心情或氛围
       - 相关的生活背景或故事（如果能推断出）
    4. 检索标签：[根据内容生成3-5个标签，包括但不限于：
       - 照片类型标签（如#自拍 #美食 #风景）
       - 场景标签（如#室内 #户外 #餐厅）
       - 活动标签（如#旅行 #聚会 #日常）
       - 情感标签（如#开心 #怀旧 #温馨）]
    
    请按照以下严格的JSON格式描述这张照片。确保输出可以被直接解析为JSON：
    {{
        "description": "这里是照片的详细描述，包含场景、人物、活动等信息。描述应该生动自然，富有感情",
        "tags": ["标签1", "标签2", "标签3"],
        "photo_type": "自拍/风景/美食/合照/截图 等分类",
        "scene_type": "室内/户外/餐厅 等场景类型",
        "emotion": "照片传达的主要情感"
    }}

    要求：
    1. description字段：用自然的语言描述照片，包含可见的关键内容和情感记忆
    2. tags字段：生成3-5个便于检索的标签，不包含#符号
    3. 如果拍摄时间和地点对描述有帮助，可以加入到描述中。但是不需要给出详细的时间和地点信息，可以用午后，早晨等描述
    4. 严格遵守JSON格式，确保输出可以被正确解析
    """
    return prompt

def address_rewrite_pt():
    return f"""
    请把给出的英文地址翻译为完整的连贯的中文地址。
    涉及到的州名请不要间简写，如CA请写成加利福尼亚州而不是加州。
    直接回复中文地址即可。
    """