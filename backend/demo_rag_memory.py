import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from tianji.knowledges.langchain_onlinellm.models import ZhipuAIEmbeddings, ZhipuLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from tianji import TIANJI_PATH
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# 加载环境变量
load_dotenv()


def create_embeddings(embedding_choice: str, cache_folder: str):
    """
    根据选择创建嵌入模型
    :param embedding_choice: 嵌入模型选择 ('huggingface' 或 'zhipuai')
    :param cache_folder: 缓存文件夹路径
    :return: 嵌入模型实例
    """
    if embedding_choice == "huggingface":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=cache_folder,
        )
    return ZhipuAIEmbeddings()


def process_psychology_data(file_path):
    """
    处理心理咨询数据集，将其转换为适合向量化的文档
    :param file_path: 数据文件路径
    :return: 文档列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 将大文本拆分成单独的问答对
    qa_pairs = []
    current_qa = {}
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith("input:"):
            # 开始新的问答对
            if current_qa and 'input' in current_qa and 'content' in current_qa:
                qa_pairs.append(current_qa)
            current_qa = {'input': line[6:].strip()}
        elif line.startswith("content:"):
            current_qa['content'] = line[8:].strip()
        elif line.startswith("reasoning_content:"):
            current_qa['reasoning'] = line[19:].strip()
    
    # 添加最后一个问答对
    if current_qa and 'input' in current_qa and 'content' in current_qa:
        qa_pairs.append(current_qa)
    
    # 创建文档列表
    documents = []
    
    for qa in qa_pairs:
        # 将每个问答对转换为一个文档
        doc_text = f"问题: {qa['input']}\n\n回答: {qa['content']}"
        if 'reasoning' in qa:
            doc_text += f"\n\n推理过程: {qa['reasoning']}"
        
        documents.append(
            Document(
                page_content=doc_text,
                metadata={"source": "psychology_dataset", "question": qa['input']}
            )
        )
    
    return documents


def create_vectordb(
    data_type: str,
    data_path: str,
    persist_directory: str,
    embedding_func,
    chunk_size: int,
    force: bool = False,
):
    """
    创建或加载向量数据库
    :param data_type: 数据类型 ('folder' 或 'web')
    :param data_path: 数据路径
    :param persist_directory: 持久化目录
    :param embedding_func: 嵌入函数
    :param chunk_size: 文本块大小
    :param force: 是否强制重建数据库
    :return: Chroma 向量数据库实例
    """
    if os.path.exists(persist_directory) and not force:
        print(f"使用现有的向量数据库: {persist_directory}")
        return Chroma(
            persist_directory=persist_directory, embedding_function=embedding_func
        )

    if force and os.path.exists(persist_directory):
        print(f"强制重建向量数据库: {persist_directory}")
        if os.path.isdir(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
        else:
            os.remove(persist_directory)

    # 分批处理文档以避免批量大小限制错误
    try:
        if data_type == "folder":
            if os.path.isdir(data_path):
                # 目录加载器
                loader = DirectoryLoader(
                    data_path, 
                    glob="*.txt", 
                    loader_cls=lambda path: TextLoader(path, encoding="utf-8")
                )
                docs = loader.load()
            else:
                # 单个文件处理
                if "psychology" in data_path.lower():
                    # 使用专门的处理函数处理心理咨询数据
                    print("检测到心理咨询数据文件，使用专门处理...")
                    docs = process_psychology_data(data_path)
                else:
                    # 普通文本文件
                    loader = TextLoader(data_path, encoding="utf-8")
                    docs = loader.load()
        elif data_type == "web":
            loader = WebBaseLoader(web_paths=(data_path,))
            docs = loader.load()
        else:
            raise gr.Error("不支持的数据类型。请选择 'folder' 或 'web'。")

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=300
        )
        split_docs = text_splitter.split_documents(docs)
        
        if len(split_docs) == 0:
            raise gr.Error("当前知识数据无效,处理数据后为空")
        
        print(f"文档总数: {len(split_docs)}")
        
        # 检查文档总数是否超过批量限制
        if len(split_docs) > 40000:  # Chroma限制略低于41666
            print(f"文档数量 {len(split_docs)} 超过最大批量, 进行分批处理")
            # 创建空数据库
            vector_db = Chroma(
                persist_directory=persist_directory, 
                embedding_function=embedding_func
            )
            
            # 分批添加文档
            batch_size = 30000  # 安全批量大小
            for i in range(0, len(split_docs), batch_size):
                end_idx = min(i + batch_size, len(split_docs))
                print(f"处理批次 {i//batch_size + 1}: 文档 {i} 到 {end_idx}")
                batch_docs = split_docs[i:end_idx]
                
                if i == 0:
                    # 第一批创建集合
                    vector_db = Chroma.from_documents(
                        documents=batch_docs,
                        embedding=embedding_func,
                        persist_directory=persist_directory,
                    )
                else:
                    # 后续批次添加到现有集合
                    vector_db.add_documents(documents=batch_docs)
        else:
            # 文档数量在限制范围内，一次性处理
            vector_db = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding_func,
                persist_directory=persist_directory,
            )
        
        return vector_db
    except Exception as e:
        raise gr.Error(f"创建向量数据库失败: {str(e)}")


def initialize_chain(
    embedding_choice: str,
    chunk_size: int,
    cache_folder: str,
    persist_directory: str,
    data_type: str,
    data_path: str,
):
    """
    初始化带有记忆功能的对话式RAG链
    :param embedding_choice: 嵌入模型选择
    :param chunk_size: 文本块大小
    :param cache_folder: 缓存文件夹路径
    :param persist_directory: 持久化目录
    :param data_type: 数据类型
    :param data_path: 数据路径
    :return: ConversationalRetrievalChain 实例
    """
    # 创建嵌入模型和向量数据库
    embeddings = create_embeddings(embedding_choice, cache_folder)
    vectordb = create_vectordb(
        data_type, data_path, persist_directory, embeddings, chunk_size, force=False
    )
    
    # 创建检索器，可以调整检索参数
    retriever = vectordb.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 4}  # 检索4个相关文档
    )
    
    # 使用ZhipuLLM
    llm = ZhipuLLM()
    
    # 创建记忆组件 - 使用ConversationBufferMemory保存对话历史
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 在链中访问历史的键名
        return_messages=True,      # 以消息格式返回历史
        output_key="answer"        # 输出结果的键名
    )
    
    # 自定义提示模板 - 系统消息
    system_prompt = """
    您是一名专业的心理咨询师，拥有丰富的临床经验和专业知识，擅长倾听、共情和提供心理支持。
    
    请根据检索到的专业心理学知识和对话历史，回应用户的心理问题。
    
    遵循以下原则:
    1. 首先表达理解和共情，让用户感到被倾听和接纳
    2. 参考之前的对话历史，保持回应的连贯性和一致性
    3. 基于检索到的心理学专业知识分析用户问题
    4. 提供分阶段的建议，包括即时缓解方法和长期策略
    5. 解释每个建议的心理学原理，使用非专业化语言
    6. 在结尾提供积极的鼓励和希望，增强用户信心
    7. 避免做出医疗诊断或替代专业治疗的建议
    8. 如果用户提到之前的对话，请准确引用并回应这些内容
    
    相关心理学知识: {context}
    对话历史: {chat_history}
    用户问题: {question}
    """
    
    # 使用自定义提示模板
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=system_prompt
    )
    
    # 创建ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,  # 返回源文档，便于调试
        verbose=True  # 打印过程信息，便于调试
    )
    
    return qa_chain


def detect_crisis(question):
    """
    检测用户问题是否包含危机信号
    :param question: 用户问题
    :return: 是否是危机情况，危机消息
    """
    crisis_keywords = [
        "自杀", "结束生命", "不想活了", "伤害自己",
        "想死", "活不下去", "绝望", "没有希望",
        "伤害他人", "报复"
    ]
    
    # 检查是否包含危机关键词
    if any(keyword in question for keyword in crisis_keywords):
        crisis_message = """
        ⚠️ 我注意到您提到了一些令人担忧的内容。如果您正在经历危机情况或有伤害自己的想法，请立即寻求专业帮助：

        - 全国心理援助热线：400-161-9995（24小时）
        - 北京危机干预中心热线：800-810-1117（24小时）
        
        请记住，任何困难都是暂时的，专业的帮助可以带来不同。您的生命珍贵且重要。
        """
        return True, crisis_message
    
    return False, ""


def handle_question(chain, question: str, chat_history):
    """
    处理用户问题，具备对话记忆功能和危机检测
    :param chain: ConversationalRetrievalChain 实例
    :param question: 用户问题
    :param chat_history: Gradio聊天界面的历史
    :return: 更新后的问题和聊天历史
    """
    if not question:
        return "", chat_history
    
    # 危机检测
    is_crisis, crisis_message = detect_crisis(question)
    
    try:
        # 使用ConversationalRetrievalChain处理问题
        # 这里链已经内置了历史记忆功能
        result = chain({"question": question})
        
        # 从结果中提取答案
        answer = result["answer"]
        
        # 在日志中打印源文档，便于调试
        if "source_documents" in result and len(result["source_documents"]) > 0:
            print(f"\n检索到的文档：")
            for i, doc in enumerate(result["source_documents"]):
                print(f"文档 {i+1}: {doc.page_content[:100]}...\n")
        
        # 如果是危机情况，在答案前添加危机消息
        if is_crisis:
            answer = f"{crisis_message}\n\n{answer}"
        
        # 更新Gradio聊天界面的历史
        chat_history.append((question, answer))
        return "", chat_history
    except Exception as e:
        error_msg = f"处理问题时出错: {str(e)}"
        print(error_msg)
        return error_msg, chat_history


def update_settings(
    embedding_choice: str,
    chunk_size: int,
    cache_folder: str,
    persist_directory: str,
    data_type: str,
    data_path: str,
):
    """
    更新设置并初始化模型，重置对话历史
    :return: 初始化的链和状态消息
    """
    try:
        chain = initialize_chain(
            embedding_choice,
            chunk_size,
            cache_folder,
            persist_directory,
            data_type,
            data_path,
        )
        # 重置记忆
        if hasattr(chain, 'memory'):
            chain.memory.clear()
        return chain, "我最近感到焦虑和压力很大，有什么方法可以缓解吗？"
    except Exception as e:
        return None, f"初始化失败：{str(e)}"


def update_data_path(data_type: str):
    """
    根据数据类型更新数据路径
    :param data_type: 数据类型
    :return: 更新后的数据路径
    """
    if data_type == "web":
        return "https://r.jina.ai/https://www.psychspace.com/psych/viewnews-20577"  # 心理健康相关网页
    
    # 直接指向心理咨询文本文件
    return "D:\\毕设\\Tianji\\try\\psychology-10k-Deepseek-R1-zh-standard.txt"


def reset_memory(chain):
    """
    重置对话记忆
    :param chain: 对话链
    :return: 状态消息
    """
    if chain is not None and hasattr(chain, 'memory'):
        chain.memory.clear()
        return "对话记忆已重置"
    return "尚未初始化模型链"


def show_memory(chain):
    """
    显示记忆状态
    :param chain: 对话链
    :return: 格式化的记忆状态
    """
    if chain is not None and hasattr(chain, 'memory'):
        history = chain.memory.chat_memory.messages
        formatted = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
        return formatted if formatted else "对话记忆为空"
    return "尚未初始化模型链"


# 创建Gradio界面
with gr.Blocks(title="心理健康咨询助手") as demo:
    gr.Markdown(
        """# 💫 心理健康咨询助手 

        这个系统基于专业心理咨询知识库，帮助您探索情绪、减轻压力、改善心理健康。
        
        提醒：
        1. 初始化数据库可能需要一些时间，请耐心等待。
        2. 如果使用过程中出现异常，将在文本输入框中显示，请不要惊慌。
        """
    )
    with gr.Row():
        embedding_choice = gr.Radio(
            ["huggingface", "zhipuai"], label="选择嵌入模型", value="huggingface"
        )
        chunk_size = gr.Slider(256, 1024, step=128, label="文本块大小", value=384)
        cache_folder = gr.Textbox(
            label="缓存文件夹路径", value=os.path.join(TIANJI_PATH, "temp")
        )
        persist_directory = gr.Textbox(
            label="向量数据库路径", value=os.path.join(TIANJI_PATH, "temp", "chromadb_hf0426")
        )
        data_type = gr.Radio(["folder", "web"], label="数据类型", value="folder")
        data_path = gr.Textbox(
            label="数据路径",
            value=os.path.join(TIANJI_PATH, "data","text_data"),
        )
        update_button = gr.Button("初始化心理咨询知识库", variant="primary")

    chatbot = gr.Chatbot(height=500, show_copy_button=True)
    msg = gr.Textbox(label="您的问题", placeholder="请描述您的心理困扰或问题...", lines=2)

    with gr.Row():
        chat_button = gr.Button("发送", variant="primary")
        clear_button = gr.ClearButton(components=[chatbot], value="清除聊天记录")
        reset_memory_button = gr.Button("重置对话记忆", variant="secondary")
    
    # 增加一个视觉分隔
    gr.Markdown("---")
    
    # 添加对话记忆状态显示（可选）
    with gr.Accordion("对话记忆状态（调试用）", open=False):
        memory_status = gr.Textbox(label="当前对话记忆", interactive=False, lines=10)
    
    # 添加心理健康资源信息
    gr.Markdown("""
    ### 💡 心理健康资源
    - **心理危机干预热线**：800-810-1117（24小时）
    - **全国心理援助热线**：400-161-9995（24小时）
    - **中国心理学会**：[https://www.cpsbeijing.org/](https://www.cpsbeijing.org/)

    *注意：本系统提供的是心理支持，不构成医疗建议。严重情况请寻求专业人士帮助。*
    """)

    data_type.change(update_data_path, inputs=[data_type], outputs=[data_path])

    model_chain = gr.State()

    update_button.click(
        update_settings,
        inputs=[
            embedding_choice,
            chunk_size,
            cache_folder,
            persist_directory,
            data_type,
            data_path,
        ],
        outputs=[model_chain, msg],
    ).then(
        show_memory,
        inputs=[model_chain],
        outputs=[memory_status]
    )
    
    # 重置记忆按钮事件
    reset_memory_button.click(
        reset_memory,
        inputs=[model_chain],
        outputs=[memory_status]
    )

    # 聊天按钮事件
    chat_button.click(
        handle_question,
        inputs=[model_chain, msg, chatbot],
        outputs=[msg, chatbot],
    ).then(
        show_memory,  # 更新记忆状态显示
        inputs=[model_chain],
        outputs=[memory_status]
    )
    
    # 回车发送
    msg.submit(
        handle_question,
        inputs=[model_chain, msg, chatbot],
        outputs=[msg, chatbot],
    ).then(
        show_memory,
        inputs=[model_chain],
        outputs=[memory_status]
    )

# 启动Gradio应用
if __name__ == "__main__":
    demo.launch()