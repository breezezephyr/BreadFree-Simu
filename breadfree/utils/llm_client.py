from openai import OpenAI
import os
import logging
from datetime import datetime

# Ensure you set this environment variable or replace it with your actual key
HUNYUAN_API_KEY = os.environ.get("HUNYUAN_API_KEY", "YOUR_API_KEY_HERE")

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
logging.basicConfig(
    filename=os.path.join(log_dir, "llm_debug.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

async def async_hunyuan_chat(
        query=None,
        prompt=None,
        model="deepseek-v3.1", # deepseek-r1 deepseek-v3.1
        temperature=0.6, 
        top_p=0.95, 
        max_tokens=4096,
        stream=False,
    ):
    try:
        # 构造 client
        client = OpenAI(
            api_key= HUNYUAN_API_KEY, # 混元 APIKey
            # base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
            base_url="https://api.lkeap.cloud.tencent.com/v1"
        )
        messages=[]
        if prompt is not None:
            messages.append({"role": "system", "content": prompt})
        if query is not None:
            messages.append({"role": "user", "content": query})
        
        # Log the request with system prompt, user query and model
        # logging.info(f"--- LLM Request ---\nModel: {model}\nPrompt: {prompt}\nQuery: {query}\n-------------------")
        # Log the request with query and model
        logging.info(f"--- LLM Request ---\nModel: {model}\nQuery: {query}\n-------------------")
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
        )
        
        response_content = completion.choices[0].message.content
        total_tokens = completion.usage.total_tokens
        
        # Log the response
        logging.info(f"--- LLM Response ---\nContent: {response_content}\nTokens: {total_tokens}\n--------------------")

        return response_content, total_tokens
    except Exception as e:
        error_msg = f"LLM Call Error: {e}"
        print(error_msg)
        logging.error(error_msg)
        return "", 0
