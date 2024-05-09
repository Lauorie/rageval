import re
import time
import pandas as pd
from tqdm import tqdm
from loguru import logger
from prompts import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

class Assessment:
    def __init__(self, model_path, excel_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)        
        self.df = pd.read_excel(excel_path)
        self.df = self.df.astype(str)  # make sure all columns are string
        self.excel_path = excel_path
    
    def get_response(self, system_prompt, user_prompt):
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(model_inputs.input_ids, max_length=8192)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
    
    def save_responses(self, assessment_types, save_path):
        start_time = time.time()
        for assessment_type in assessment_types:
            logger.info(f"Processing {assessment_type}...")
            scores, responses = self.get_scores(assessment_type)
            self.df[f'{assessment_type}_score'] = scores
            self.df[f'{assessment_type}_response'] = responses
        end_time = time.time()
        self.df.to_excel(save_path, index=False)
        logger.info("All assessments have been processed and saved, time elapsed: {:.2f}/3600 hours".format((end_time - start_time) / 3600))
        
    def get_scores(self, assessment_type):
        prompts = {
            'groundedness': (Prompt.GROUNDEDNESS_SYSTEM_PROMPT, Prompt.GROUNDEDNESS_USER_PROMPT, ['answer', 'ground_truth']),
            'context_recall': (Prompt.CONTEXT_RECALL_SYSTEM_PROMPT, Prompt.CONTEXT_RECALL_USER_PROMPT, ['contexts', 'ground_truth']),
            'context_precision': (Prompt.CONTEXT_PRECISION_SYSTEM_PROMPT, Prompt.CONTEXT_PRECISION_USER_PROMPT, ['question', 'contexts']),
            'faithfulness': (Prompt.FAITHFULNESS_SYSTEM_PROMPT, Prompt.FAITHFULNESS_USER_PROMPT, ['contexts', 'answer']),            
            'answer_relevance': (Prompt.ANSWER_RELEVANCE_SYSTEM_PROMPT, Prompt.ANSWER_RELEVANCE_USER_PROMPT, ['question', 'answer']),           
        }
        system_prompt, user_prompt_template, required_columns = prompts[assessment_type]
        scores = []
        responses = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df),desc=f"{assessment_type}"):
            user_prompt = user_prompt_template.format(**{col: row[col] for col in required_columns})
            response = self.get_response(system_prompt, user_prompt)
            scores.append(self.calculate_average_score(response))
            responses.append(response)
        return scores, responses
   
    @staticmethod
    def calculate_average_score(response):
        if isinstance(response, list):
            scores = [int(r['score']) for r in response]
        else:
            scores = [int(score) for score in re.findall(r'"score": (\d+)', response)]
        return round(sum(scores) / (len(scores) * 10), 2) if scores else 0

# example
if __name__ == '__main__':
    model_path = "/root/app/models/Qwen1.5-32B-Chat"
    excel_path = "/root/app/PLC问题测试.xlsx"
    save_path = "/root/app/PLC问题测试-评估.xlsx"
    assessment_types = ['groundedness', 'context_recall', 'context_precision', 'faithfulness', 'answer_relevance']
    assessment = Assessment(model_path, excel_path)
    assessment.save_responses(assessment_types, save_path)
    
    
        





