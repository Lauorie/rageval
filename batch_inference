mport os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import json
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"  # the device to load the model onto

model_path = '/root/app/models/Qwen1.5-32B-Chat'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left') # 72时设置 padding_side='left'

def get_responses(system_message, user_messages, batch_size=4):
    messages_batch = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        for user_message in user_messages
    ]
    texts = [tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True
    ) for msg in messages_batch]

    responses = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Responses"):
        batch_texts = texts[i:i + batch_size]
        model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        responses.extend(batch_responses)

    return responses
  trans_system_prompt_ = 'You are an expert in translation.'
trans_user_prompt_ = 'Please translate the following text into Chinese:\ n'

summary_system_prompt_ = 'You are an expert in text summarization.'
summary_user_prompt_ = 'Please generate a concise summary for the following text:\ n'

json_dir = '/root/app/workspace/trans'
json_files = os.listdir(json_dir)
json_files = [file for file in json_files if file.endswith('.json')]

for json_file in json_files:
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    print(f'Processing {json_file}...')
    batch_summaries = get_responses(trans_system_prompt_, [trans_user_prompt_ + text for text in tqdm(texts, desc="Preparing Inputs")])

    res = []
    for item, summary in zip(data, batch_summaries):
        json_data = {
            "id": item['id'],
            "conversations": [
                {
                    "from": "user",
                    "value": trans_user_prompt_ + item['text']
                },
                {
                    "from": "assistant",
                    "value": summary
                }
            ],
            "data_type": "RAG_data",
            "subject": "abstracton",
            "language": "CHN",
            "domain": "",
            "turn_num": 1
        }
        res.append(json_data)
    
    with open(json_path[:-5] + '_trans.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    
    print(f'{json_file} done!')

print('All done!')
