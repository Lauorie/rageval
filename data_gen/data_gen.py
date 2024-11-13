# gen_rag_data.py
import os
import json
import time
import pandas as pd
from tqdm import tqdm
import re
from loguru import logger
from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain_core.documents.base import Document


class CustomOpenAI(OpenAI):
    """自定义OpenAI类,用于生成数据"""
    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__(api_key=api_key, base_url=base_url)
        self.model = model
        
    def invoke(self, prompt, max_retries=3, delay=2):
        for attempt in range(max_retries):
            try:
                response = self.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"第{attempt + 1}次调用API失败: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"达到最大重试次数,跳过该样本")
                    return None
                time.sleep(delay)
                continue

class PDFProcessor:
    """PDF处理类"""
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        
    def load_and_split(self, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
        """加载PDF并分割文本"""
        # Load PDF documents
        loader = PyPDFDirectoryLoader(self.pdf_dir)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；"],
        )
        
        docs = text_splitter.split_documents(documents)
        
        # Filter out short documents
        docs = [doc for doc in docs if len(doc.page_content) > 10]
        
        return docs, documents

class QAGenerator:
    """问答数据生成类"""
    def __init__(self, llm: CustomOpenAI):
        self.llm = llm
        self.load_prompts()
        
    def load_prompts(self):
        """加载提示模板"""
        self.question_prompt = PromptTemplate(
            input_variables=["context"],
            template="""
            <Instructions>
            Here are some examples:
            <examples>
            question:人脸识别属于以下哪种类型：强人工智能、弱人工智能、超人工智能？
            type:分类

            question:从中金公司纰漏的信息中抽取融资相关信息： - 公司名称 - 融资轮次 - 融资金额 - 投资方 - 资金用途 
            type:信息抽取

            question:哪篇论文研究了 RAG 领域中不同 chunk_size 的表现？
            type:文档定位

            question:贞观之治的统治者是康熙，对吗？
            type:事实判断

            question:请分析 Nvidia 和 Apple 过去三年的财务表现,并判断哪家公司更值得投资
            type:多步推理

            question:朗新科技2023年上半年的限制性股票激励计划授予的股票数量与年初相比有何变化？
            type:数据对比
            </examples>

            Your task is to choose one of the types above, and generate 1 question that can be answered using the provided context, following these rules:

            <rules>
            1. The question should make sense to humans even when read without the given context.  
            2. The question should be fully answered from the given context.
            3. The question should be framed from a part of context that contains important information. It can also be from tables, code, etc.
            4. The answer to the question should not contain any links.
            5. The question should be of moderate difficulty.
            6. The question must be reasonable and must be understood and responded by humans.
            7. Do not use phrases like 'provided context', etc. in the question.
            8. The question should not contain more than 30 words, make use of abbreviations wherever possible.
            9. The question should be in Chinese.
            </rules>

            Provide your answer in JSON format:
            ```json
            {{
              "question": "question generated",
              "type": "type chosen from the examples"
            }}
            ```

            Here is some context:
            <context>
            {context}
            </context>

            </Instructions>
            """
        )
        
        self.answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            <Instructions>
            <Task> 
            <role>You are an experienced QA Engineer for building large language model applications.</role>
            <task>It is your task to generate an answer to the following question <question>{question}</question> only based on the <context>{context}</context></task>
            
            <rules>
            1. Only use the given context as a source for generating the answer.
            2. Be as precise as possible with answering the question.
            3. Be concise in answering the question and only answer the question at hand rather than adding extra information.
            4. The answer should be in Chinese.
            </rules>

            Only output the generated answer. No extra characters.
            </Task>
            </Instructions>
            """
        )

        self.question_compress_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            <Instructions>
            <role>You are an experienced linguistics expert for building testsets for large language model applications.</role>
            
            <task>It is your task to rewrite the following question in a more indirect and compressed form, following these rules:
            
            <rules>
            1. Make the question more indirect
            2. Make the question shorter  
            3. Use abbreviations if possible
            4. Make the question more user-friendly
            </rules>

            <question>
            {question}
            </question>

            Output only the rewritten question. Do not provide any other explanation or text.
            </task>
            </Instructions>
            """
        )

    def generate_qa_pair(self, doc: Document) -> Dict:
        """生成问答对"""
        # Generate question
        question_prompt = self.question_prompt.format(context=doc.page_content)

        question_response = self.llm.invoke(question_prompt)
        
        # If the question generation fails, return None
        if not question_response:
            return None
            
        try:
            question = json.loads(question_response.strip('```json').strip('```').strip())
        except:
            logger.error("JSON parsing failed")
            return None
            
        time.sleep(0.5)

        # Generate compressed question
        compressed_question = self.llm.invoke(
            self.question_compress_prompt.format(question=question["question"])
        )
        time.sleep(0.5)

        # Generate answer
        answer = self.llm.invoke(
            self.answer_prompt.format(question=question["question"], context=doc.page_content)
        )
        time.sleep(0.5)
        
        return {
            "question": question["question"],
            "question_compressed": compressed_question, 
            "type": question["type"],
            "reference_answer": answer,
            "source_raw": doc.page_content,
            "source_document": doc.metadata["source"]
        }

class Evaluator:
    """评估类"""
    def __init__(self, llm: CustomOpenAI):
        self.llm = llm
        self.load_prompts()
        
    def load_prompts(self):
        """加载评估提示模板"""
        self.groundedness_prompt = PromptTemplate(
            input_variables=["context","question"],
            template="""
            <Instructions>
            You will be given a context and a question related to that context.

            Your task is to provide an evaluation of how well the given question can be answered using only the information provided in the context. Rate this on a scale from 1 to 5, where:

            1 = The question cannot be answered at all based on the given context
            2 = The context provides very little relevant information to answer the question  
            3 = The context provides some relevant information to partially answer the question
            4 = The context provides substantial information to answer most aspects of the question
            5 = The context provides all the information needed to fully and unambiguously answer the question

            First, read through the provided context carefully:

            <context>
            {context}
            </context>

            Then read the question:

            <question>
            {question}  
            </question>

            <rules>The evaluation should be in Chinese.</rules>
                 
            Evaluate how well you think the question can be answered using only the context information. Provide your reasoning first in an <evaluation> section, explaining what relevant or missing information from the context led you to your evaluation score in only one sentence.

            Provide your evaluation in the following format:

            <rating>(Your rating from 1 to 5)</rating>
            
            <evaluation>(Your evaluation and reasoning for the rating)</evaluation>
            
            </Instructions>
            """
        )
        
        self.relevance_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            <Instructions>
            You will be given a question related to 小米SU7用户手册. Your task is to evaluate how useful this question would be for a customer who is interested in 小米SU7.

            To evaluate the usefulness of the question, consider the following criteria:

            1. Relevance: Is the question directly relevant to your work? Questions that are too broad or unrelated to this domain should receive a lower rating.

            2. Practicality: Does the question address a practical problem or use case that analysts might encounter? Theoretical or overly academic questions may be less useful.

            3. Clarity: Is the question clear and well-defined? Ambiguous or vague questions are less useful.

            4. Depth: Does the question require a substantive answer that demonstrates understanding of financial topics? Surface-level questions may be less useful.

            5. Applicability: Would answering this question provide insights or knowledge that could be applied to real-world company evaluation tasks? Questions with limited applicability should receive a lower rating.

            <rules>The evaluation should be in Chinese.</rules>

            Provide your evaluation in the following format:

            <rating>(Your rating from 1 to 5)</rating>

            <evaluation>(Your evaluation and reasoning for the rating)</evaluation>

            Here is an example:
            <evaluation>The question is very relevant to the persona because it asks about financial information of a company</evaluation> 
            <rating>5</rating>

            Here is the question:

            {question}
            </Instructions>
            """
        )
    
    def evaluate(self, question: str, context: str) -> Dict:
        """评估问题"""
        # Check groundedness
        groundedness = self.llm.invoke(
            self.groundedness_prompt.format(question=question, context=context)
        )
        groundedness_score = self.extract_rating(groundedness)
        groundedness_reasoning = self.extract_reasoning(groundedness)
        
        # Check relevance
        relevance = self.llm.invoke(
            self.relevance_prompt.format(question=question)
        )
        relevance_score = self.extract_rating(relevance)
        relevance_reasoning = self.extract_reasoning(relevance)
        
        return {
            "groundedness_score": groundedness_score,
            "groundedness_reasoning": groundedness_reasoning,
            "relevance_score": relevance_score, 
            "relevance_reasoning": relevance_reasoning
        }
        
    @staticmethod
    def extract_rating(text: str) -> Optional[int]:
        """提取评分"""
        pattern = r'<rating>(.*?)</rating>'
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
        return None
    
    @staticmethod  
    def extract_reasoning(text: str) -> Optional[str]:
        """提取理由"""
        pattern = r'<evaluation>(.*?)</evaluation>'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

def main():
    # Initialize DeepSeek
    # api_key = "sk-bba63edb9a2545a2b17329c42714"
    # base_url = "https://api.deepseek.com"
    # model = "deepseek-chat"

    # Initialize OpenAI
    api_key = "sk-Kdu5mlKMlLx2QBWdA940E93a4e638c7e6b2d1034C62e"
    base_url = "https://one-api.com/v1"
    model = "gpt-4o-2024-08-06"

    # Initialize XAI
    # api_key = "xai-eUy22SMHpaMbGM5M5kHqAmcQVCYxKEqlvWBJCgugk3LqFSJmKu4Hu93otjmMNUttXICadqKV"
    # base_url = "https://api.x.ai/v1"
    # model = "grok-beta"
    
    llm = CustomOpenAI(api_key=api_key, base_url=base_url, model=model)
    
    # Process PDF
    pdf_dir = "//root/app/google_pdf/split_pdfs/folder_4"
    pdf_processor = PDFProcessor(pdf_dir)
    docs, raw_documents = pdf_processor.load_and_split()
    
    logger.info(f'已加载 {len(raw_documents)} 页文档,平均每页字符数为 {sum([len(doc.page_content) for doc in raw_documents])//len(raw_documents)}。')
    logger.info(f'分割后共有 {len(docs)} 块文档。')
    logger.info(f'分割后的 {len(docs)} 块文档平均字符数为 {sum([len(doc.page_content) for doc in docs])//len(docs)}。')
    
    # Generate QA pairs
    qa_generator = QAGenerator(llm)
    dataset = []
    SAVE_INTERVAL = 10  # 每处理10个文档保存一次
    
    # # Evaluate dataset
    # # evaluator = Evaluator(llm)
    # # for index, row in dataset_df.iterrows():
    # #     evaluation = evaluator.evaluate(row["question"], row["source_raw"])
    # #     for key, value in evaluation.items():
    # #         dataset_df.at[index, key] = value
            

    try:
        for i, doc in enumerate(tqdm(docs, desc="Generating QA pairs")):
            try:
                qa_pair = qa_generator.generate_qa_pair(doc)
                if qa_pair:
                    dataset.append(qa_pair)
                
                # 定期保存数据
                if (i + 1) % SAVE_INTERVAL == 0:
                    temp_df = pd.DataFrame(dataset)
                    temp_save_path = "/root/app/google_pdf/split_pdfs/folder_4_temp.json"
                    temp_df.to_json(temp_save_path, orient='records', force_ascii=False, indent=4)
                    logger.info(f"已临时保存 {len(dataset)} 条数据到 {temp_save_path}")
                    
            except Exception as e:
                logger.error(f"处理第 {i+1} 个文档时出错: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"主程序出错: {str(e)}")
    finally:
        # 确保最终数据被保存
        if dataset:
            try:
                final_df = pd.DataFrame(dataset)
                save_path = "/root/app/google_pdf/folder_4_train.json"
                final_df.to_json(save_path, orient='records', force_ascii=False, indent=4)
                logger.info(f"最终保存了 {len(dataset)} 条数据到 {save_path}")
            except Exception as e:
                logger.error(f"保存最终数据时出错: {str(e)}")
                # 进行备份保存尝试
                try:
                    backup_path = "/root/app/google_pdf/folder_4_backup_train.json"
                    json.dump(dataset, open(backup_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
                    logger.info(f"数据已备份到 {backup_path}")
                except:
                    logger.error("备份保存也失败了!")

if __name__ == "__main__":
    main()