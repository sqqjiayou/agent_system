import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from typing import Optional, Dict, Any

# 设置日志记录
logger = logging.getLogger('api_calls')
logger.setLevel(logging.DEBUG)

# 移除所有现有的处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 设置文件处理器
log_file = os.path.join(log_dir, f'api_calls_{time.strftime("%Y%m%d")}.log')
print(f"Creating log file at: {log_file}")

try:
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.DEBUG)
    print("Successfully created file handler")
except Exception as e:
    print(f"Error creating file handler: {str(e)}")

# 设置控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 立即测试日志记录
logger.debug("Logger initialization completed")
logger.info("API logging system started")

# 状态图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "⟳"


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 验证环境变量
api_key = os.getenv("DS_API_KEY")
model = os.getenv("DS_MODEL")

if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 DS_API_KEY 环境变量")
    raise ValueError("DS_API_KEY not found in environment variables")
if not model:
    model = "deepseek-chat"
    logger.info(f"{WAIT_ICON} 使用默认模型: {model}")

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://tbnx.plus7.plus/v1"
)
logger.info(f"{SUCCESS_ICON} DeepSeek 客户端初始化成功")

@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=600,
    giveup=lambda e: "Rate limit" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """带重试机制的内容生成函数"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 DeepSeek API...")
        logger.info(f"请求内容: {contents[:500]}..." if len(
            str(contents)) > 500 else f"请求内容: {contents}")
        logger.info(f"请求配置: {config}")

        messages = [{"role": "user", "content": contents}]
        if config and 'system_instruction' in config:
            messages.insert(0, {"role": "system", "content": config['system_instruction']})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            timeout=180,  # 添加180秒的超时设置
            request_timeout=180,  # 请求超时时间
        )

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        response_text = response.choices[0].message.content
        logger.info(f"响应内容: {response_text[:500]}..." if len(
            str(response_text)) > 500 else f"响应内容: {response_text}")
        
        # 创建一个与之前格式兼容的响应对象
        class Response:
            def __init__(self, text):
                self.text = text
        
        return Response(response_text)
    except Exception as e:
        if "Rate limit" in str(e):
            logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {str(e)}")
            time.sleep(5)
            raise e
        logger.error(f"{ERROR_ICON} API 调用失败: {str(e)}")
        logger.error(f"错误详情: {str(e)}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """获取聊天完成结果，包含重试逻辑"""
    try:
        if model is None:
            model = os.getenv("DS_MODEL", "deepseek-chat")

        logger.info(f"{WAIT_ICON} 使用模型: {model}")
        logger.debug(f"消息内容: {messages}")

        for attempt in range(max_retries):
            try:
                # 转换消息格式
                prompt = ""
                system_instruction = None

                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        system_instruction = content
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"

                # 准备配置
                config = {}
                if system_instruction:
                    config['system_instruction'] = system_instruction

                # 调用 API
                response = generate_content_with_retry(
                    model=model,
                    contents=prompt.strip(),
                    config=config
                )

                if response is None:
                    logger.warning(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)  # 使用3作为基数
                        retry_delay = min(retry_delay, 30)  # 最大等待时间30秒
                        # retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    return None

                # 转换响应格式
                chat_message = ChatMessage(content=response.text)
                chat_choice = ChatChoice(message=chat_message)
                completion = ChatCompletion(choices=[chat_choice])

                logger.debug(f"API 原始响应: {response.text}")
                logger.info(f"{SUCCESS_ICON} 成功获取响应")
                return completion.choices[0].message.content

            except Exception as e:
                logger.error(
                    f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (3 ** attempt)  # 使用3作为基数
                    retry_delay = min(retry_delay, 120)  # 最大等待时间120秒
                    # retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                    return None

    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None