import requests
import base64
import re  # 导入正则模块，用于过滤标记

def get_poker_suit(image_path, api_key):
    """
    识别扑克牌花色（新增标记过滤逻辑）
    :param image_path: 本地扑克牌图片路径
    :param api_key: 智谱AI的API密钥（格式："Bearer xxxxxx"）
    :return: 花色结果（红桃/黑桃/梅花/方块/错误信息）
    """
    # 1. 将本地图片转换为base64编码
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        return f"图片处理失败：{str(e)}"

    # 2. 构造请求参数
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "model": "glm-4.5v",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_str}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "仅识别这张扑克牌的花色，直接返回结果（红桃/黑桃/梅花/方块），不要其他任何内容。"
                    }
                ]
            }
        ]
    }

    # 3. 发送请求并解析结果（新增标记过滤）
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        suit = result["choices"][0]["message"]["content"].strip()
        
        # 关键修改：过滤智谱模型的特殊标记（正则匹配并移除）
        suit = re.sub(r'<\|begin_of_box\|>|<\|end_of_box\|>', '', suit).strip()
        
        # 验证结果是否合法
        if suit in ["红桃", "黑桃", "梅花", "方块"]:
            return suit
        else:
            return f"识别结果异常：{suit}"
    except requests.exceptions.HTTPError as e:
        return f"接口请求失败：{str(e)}，详情：{response.text}"
    except KeyError as e:
        return f"解析结果失败：缺少字段{e}"
    except Exception as e:
        return f"其他错误：{str(e)}"

if __name__ == "__main__":
    # 配置参数（替换为你的实际信息）
    POKER_IMAGE_PATH = "/Users/bytedance/Desktop/bishe/data/t1.jpg"
    API_KEY = "Bearer 90c73c1e514a4f8bb95641308e80194d.XdI2BrRv6eoV1nUY"

    result = get_poker_suit(POKER_IMAGE_PATH, API_KEY)
    print(result)