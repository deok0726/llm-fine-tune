import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from docx import Document

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
# model_id = 'Bllossom/llama-3-Korean-Bllossom-70B'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

def read_word_file(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            full_text.append(paragraph.text)
    return "\n".join(full_text)

# 텍스트 요약 함수
def summarize_text(text):
    document_text = read_word_file(file_path)
    PROMPT = '''Your job is to identify the key points of the contract and summarize them. Follow the instructions below to summarize the contract.당신의 역할은 계약서의 핵심 내용을 파악하고, 이를 요약하는 것입니다. 아래 지시사항을 따라서 계약서 요약을 수행합니다.'''
    instruction = f'''업무위탁계약서 파일을 보고, 계약 내용 중 핵심 내용들을 요약해주세요. 
                    - '계약 날짜', '총 계약 기간', '계약 금액', '계약 대상 업무(위탁 업무)', '서비스 수준'은 핵심 내용이므로, 답변에 반드시 포함해주세요.
                    - '총 계약 기간'은 '계약 날짜'부터 시작해야 합니다. 총 계약 기간은 다음과 같은 형식으로 답변해주세요. (ex. 총 계약 기간: 2024.01.02~2025.01.01)
                    - '서비스 수준'은 서비스 별 상세내용과 제공횟수에 대한 내용이 포함되어야 합니다.
                    
                    업무위탁계약서: {document_text}
                    요약: '''
    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.5,
        repetition_penalty = 1.1
    )
    
    summary = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return summary

if __name__ == "__main__":
    file_path = "01_업무위탁계약서(표준)2024 수정_비식별화 1.docx"
    text = read_word_file(file_path)

    summary = summarize_text(text)
    print("요약 내용:")
    print(summary)
