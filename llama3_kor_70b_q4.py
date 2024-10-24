import sys
from llama_cpp import Llama
from transformers import AutoTokenizer

model_id = 'Bllossom/llama-3-Korean-Bllossom-70B-gguf-Q4_K_M'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Llama(
    model_path='/svc/project/llm-fine-tune/model/llama-3-Korean-Bllossom-70B-gguf-Q4_K_M.gguf',
    # n_ctx=512,
    n_gpu_layers=-1
)

max_n_ctx = model.config['max_position_embeddings']  # 이 부분은 실제로는 모델에 따라 다를 수 있음
print(f"Maximum n_ctx for this model is: {max_n_ctx}")

sys.exit()

PROMPT = \
'''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''

instruction = 'Your Instruction'

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize = False,
    add_generation_prompt=True
)

generation_kwargs = {
    "max_tokens":512,
    "stop":["<|eot_id|>"],
    "echo":True, # Echo the prompt in the output
    "top_p":0.9,
    "temperature":0.6,
}

resonse_msg = model(prompt, **generation_kwargs)
print(resonse_msg['choices'][0]['text'][len(prompt):])
