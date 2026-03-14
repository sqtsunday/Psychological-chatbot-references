import os
import sys
import logging
from dotenv import load_dotenv
from src.shared.schema_extraction import schema_extraction_from_text

# 加载环境变量
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 临时添加zhipu模型映射到MODEL_VERSIONS字典
from src.llm import MODEL_VERSIONS
if "zhipu/glm-4" not in MODEL_VERSIONS:
    MODEL_VERSIONS["zhipu/glm-4"] = "glm-4"

# 测试文本示例
test_text = """
adhd的治疗方法，焦虑和adhd的因果关系，减轻焦虑的方法
"""

# 选择模型类型
model_type = "zhipu/glm-4"

# 确保zhipu的API密钥已设置
if not os.getenv("ZHIPUAI_API_KEY"):
    raise ValueError("未找到ZHIPUAI_API_KEY环境变量")

# 测试两种模式
try:
    print("使用WITH_SCHEMA模式提取：")
    result1 = schema_extraction_from_text(test_text, model_type, True)
    print(f"提取的节点类型: {result1.labels}")
    print(f"提取的关系类型: {result1.relationshipTypes}")

    print("\n使用WITHOUT_SCHEMA模式提取：")
    result2 = schema_extraction_from_text(test_text, model_type, False)
    print(f"提取的节点类型: {result2.labels}")
    print(f"提取的关系类型: {result2.relationshipTypes}")
except Exception as e:
    print(f"执行过程中发生错误: {e}")
    import traceback
    traceback.print_exc()