import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from docx import Document

model_id = 'Bllossom/llama-3-Korean-Bllossom-70B'
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


def summarize_document(file_path):
    document_text = read_word_file(file_path)
    
    PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
    instruction = f"입력된 문서의 내용을 요약해줘. 문서 내용: {document_text}"
    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.3,
        top_p=0.8
    )
    print("output finished")

    summary = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    file_path = "01_업무위탁계약서(표준)2024 수정_비식별화 1.docx"
    summary = summarize_document(file_path)

    # 요약 내용 출력
    print("요약 내용:")
    print(summary)
