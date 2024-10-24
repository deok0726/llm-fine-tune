import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from docx import Document

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

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

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                full_text.append(stripped_line)
    return "\n".join(full_text)

def summarize_contract(text):
    with open("/svc/project/llm-fine-tune/output/contract_8b_kor.txt", 'w') as f:
        document_text = read_word_file(text)
        prompt = '''# 역할
                    당신의 역할은 계약서의 핵심 내용을 파악하고, 이를 요약하는 것입니다. 아래 지시사항을 따라서 계약서 요약을 수행합니다.  '''
        instruction = f'''# 지시사항
                            1단계) 업무위탁계약서 파일을 보고, 계약 내용 중 핵심 내용들을 요약해주세요. 
                            - '계약 날짜', '총 계약 기간', '계약 금액', '계약 대상 업무(위탁 업무)', '서비스 수준', '계약 시 계약 이행을 위해 1) 수탁자에게 제공해야 하는 신용정보의 리스트와 2) 수탁자에게 제공받아야 하는 신용정보의 리스트'은 핵심 내용이므로, 답변에 반드시 포함해주세요.
                            - '총 계약 기간'은 '계약 날짜'부터 시작해야 합니다. 총 계약 기간은 다음과 같은 형식으로 답변해주세요. (ex. 총 계약 기간: 2024.01.02~2025.01.01)
                            - '서비스 수준'은 서비스 별 상세내용과 제공횟수에 대한 내용이 요약에 포함되어야 합니다.
                            -'신용정보' 내용을 요약할 때는 만약 해당 내용이 없다면, "본 계약서에는 해당 내용이 없습니다."라고 답변해주세요. 

                            2단계) 1단계에서 요약한 내용을 보기 쉽게 표 형태로 정리해주세요.
                            - 요약 내용을 표로 정리할 때는 내용이 변형, 누락되지 않고 그대로 들어가야 합니다.

                            Think step by step. 
                            Answer in Korean.
                                                    
                            업무위탁계약서: {document_text}
                            요약: '''
        messages = [
            {"role": "system", "content": f"{prompt}"},
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
            max_new_tokens=4096,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.5,
            repetition_penalty = 1.1
        )
        
        summary = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        print("LLM 답변: \n", summary)
        f.write(summary + "\n\n")

    return summary

def renew_contract(path_22, path_24):
    torch.cuda.empty_cache()
    with open("/svc/project/llm-fine-tune/output/renew_8b_kor_3.txt", 'w') as f:
        document_text_22 = read_word_file(path_22)
        document_text_24 = read_word_file(path_24)
        prompt = f'''
                두 문서는 회사간 계약 내용에 대한 내용을 담고 있습니다. 
                ‘23년 계약서'는 '24년 계약서'의 갱신 계약입니다.
                당신의 역할은 두 문서의 내용을 분석하고, 내용을 비교 분석하는 것입니다. 
                구체적인 업무는 다음 지시사항을 따르세요.

                - 갱신 계약서와 기존 계약서의 내용과 다른 부분을 찾아내고, 그 내용이 몇 조에 해당하는 내용인지 알려주세요. 
                - 답변할 때는 '갱신 계약시 달라진 내용의 원문', '기존 계약의 내용 원문'을 모두 알려주세요. 
                - 차이점이 있는 부분은 모두 식별해서 알려주시기 바랍니다. 
                - 추가로 변경된 내용이 있을 때 그 내용이 '롯데카드' 측에 유리한 내용인지 혹은 불리한 내용인지 판단하고, 그 근거 또한 설명해주세요. 
                - 답변은 아래 형식을 따라서 해주세요.

                [답변형식 예시]

                두 계약서 ('22년 임대차계약서'와 ‘24년 임대차계약서')의 차이점을 다음과 같이 분석했습니다.
                1) 계약금액 (제3조)
                - 갱신 계약 (‘24년 계약서): "총 계약 금액 : 일금 147,242,500원/ VAT별도" 
                - 기존 계약 (‘24년 계약서): "총 계약 금액 : 일금 143,867,500원/ VAT별도" 
                - 변경 내용 요약: 총 계약 금액이 약 3,375,000원 증가하였고, PC 백신의 가격이 증가했습니다.
                - 판단: 이 내용은 롯데카드 측에 유리한 내용입니다. 총 계약 금액이 올랐습니다.

                ### 아래 입력 파일을 기반으로 양식에 맞게 답해주세요.

                24년 계약서: {document_text_22}
                23년 계약서: {document_text_24}
                
                답변: '''

        # input_ids = tokenizer(
        #     prompt,
        #     return_tensors="pt",
        #     # max_length=model.config.max_position_embeddings*2,
        #     # truncation=True,
        # ).to(model.device)

        # terminators = [
        #     tokenizer.eos_token_id,
        #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

        # outputs = model.generate(
        #     input_ids,
        #     max_new_tokens=1024,
        #     eos_token_id=terminators,
        #     do_sample=True,
        #     temperature=0.1,
        #     top_p=0.5,
        #     repetition_penalty=1.1
        # )

        # summary = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.5,
            repetition_penalty=1.1
        )
        
        # summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        print("LLM 답변: \n", summary)
        
        f.write(summary + "\n\n")

def new_contract(a, b, c):
    torch.cuda.empty_cache()
    with open("/svc/project/llm-fine-tune/output/new_contract_8b_kor.txt", 'w') as f:
        document_text_a = read_word_file(a)
        document_text_b = read_word_file(b)
        document_text_c = read_word_file(c)
        prompt = f'''
                업로드된 문서들은 특정 계약 내용을 포함한 계약서입니다.  
                당신의 역할은 각 계약서의 내용을 분석하고, 롯데카드와 A사 간 계약 내용을 롯데카드와 다른 회사들(B사, C사) 간 계약 내용과 비교하는 것입니다. 
                계약서의 종류는 크게 2가지로 구분됩니다. 
                - 표준위탁계약서 : 롯데카드와 다른 회사의 계약 상 관계를 판단할 수 있는 계약서
                - 각 붙임 계약서 : 롯데카드와 A사/B사/C사 간 계약서 내용 차이를 파악하기 위한 계약서

                다음 지시사항에 따라 계약서를 비교분석하세요.
                - 롯데카드와 A사 간 계약서에서 롯데카드와 나머지 회사들(B사, C사) 간 계약서와 차이점이 있는 부분을 찾아내고, 해당 내용이 몇 조에 해당하는지 명시하세요.
                - 차이점이 있는 경우, '롯데카드와 나머지 회사들과의 계약 내용'을 기준으로 '롯데카드와 A사 간 계약이 롯데카드에게 유리한지/불리한지' 판단하세요. 유리/불리한지 판단할 때는 첨부한 표준 계약서의 내용 내 롯데카드와 A사/B사/C사의 관계에 대해서 분석한 결과를 기반으로 판단해주세요.    
                - 유리한 경우: 롯데카드가 더 적은 비용을 부담하거나, 더 많은 혜택을 받는 경우를 의미합니다. 예를 들어, 수수료율이 낮거나, 롯데카드가 부담해야 할 의무가 적을 때 유리하다고 판단합니다. 
                - 불리한 경우: 롯데카드가 더 많은 비용을 지불하거나, 더 불리한 조건에 처하는 경우를 의미합니다. 예를 들어, 수수료율이 높거나, 롯데카드가 부담해야 할 의무가 더 클 때 불리하다고 판단합니다.
                - 답변 시 A사와의 계약 내용과 나머지 회사들(B사, C사)과의 계약 내용을 각각 제시하세요. 계약서 파일이 여러 개이므로, B사와 C사 간 계약 내용을 명확히 구분하여 작성하세요. 계약 기간에 대한 차이점은 제외하고, 나머지 차이점만을 식별하여 분석하세요.

                [답변 형식 예시]
                롯데카드 - A사 간 계약서 내용과 타 계약서 내용의 차이점을 분석한 결과입니다.
                1. 임대차 기간 (제4조)
                - A사와의 계약 : "임대차기간은 2024년 1월 1일부터 2025년 12월 31일까지로 한다.“
                - 분석: 임대차 기간에 차이가 있지만, 해당 조건이 롯데카드의 비용 부담이나 혜택에 직접적인 영향을 미치지 않으므로 유/불리 판단에서 제외합니다.
                - 중요: 유리한 경우와 불리한 경우의 기준은 롯데카드가 실질적으로 더 적은 비용을 부담하거나 혜택을 얻는지에 달려 있습니다.


                A사 계약서: {document_text_a}
                B사 계약서: {document_text_b}
                C사 계약서: {document_text_c}
                
                답변: '''

        # input_ids = tokenizer(
        #     prompt,
        #     return_tensors="pt",
        #     # max_length=model.config.max_position_embeddings*2,
        #     # truncation=True,
        # ).to(model.device)

        # terminators = [
        #     tokenizer.eos_token_id,
        #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

        # outputs = model.generate(
        #     input_ids,
        #     max_new_tokens=1024,
        #     eos_token_id=terminators,
        #     do_sample=True,
        #     temperature=0.1,
        #     top_p=0.5,
        #     repetition_penalty=1.1
        # )

        # summary = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # 어텐션 마스크 전달
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.5,
            repetition_penalty=1.1
        )
        
        # summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        print("LLM 답변: \n", summary)
        
        f.write(summary + "\n\n")

if __name__ == "__main__":
    # summarize_contract("/svc/project/llm-fine-tune/data/contract/01_업무위탁계약서(표준)2024 수정_비식별화.docx")

    # path_22 = "/svc/project/llm-fine-tune/data/contract/(1) 임대차 계약서/① '22년 임대차계약서.docx"
    # path_24 = "/svc/project/llm-fine-tune/data/contract/(1) 임대차 계약서/① '24년 임대차계약서.docx"

    # path_22 = "/svc/project/llm-fine-tune/data/contract/(2) 라이선스 계약서/② '22년 v3백신 및 pms 라이선스 갱신계약 _3. 계약서.docx"
    # path_24 = "/svc/project/llm-fine-tune/data/contract/(2) 라이선스 계약서/② '23년 v3백신 및 pms 라이선스 갱신계약 _[검토] 3. 계약서.docx"
    
    path_22 = "/svc/project/llm-fine-tune/data/contract/(3) 채권추심 위임 계약서/③ 23년 4분기 채권추심 위임 계약_1.표준위탁계약서_붙임.docx"
    path_24 = "/svc/project/llm-fine-tune/data/contract/(3) 채권추심 위임 계약서/③ 24년 1분기 채권추심 위임 계약_1.표준위탁계약서_붙임.docx"
    
    renew_contract(path_22, path_24)

    # a = "/svc/project/llm-fine-tune/data/contract/(4) 신규 계약서/① 1.표준위탁계약서_A사(2024년 하반기)_붙임.docx"
    # b = "/svc/project/llm-fine-tune/data/contract/(4) 신규 계약서/② 3.표준위탁계약서_B사(2024년)_붙임.docx"
    # c = "/svc/project/llm-fine-tune/data/contract/(4) 신규 계약서/③ 3.표준위탁계약서_C사(2024년)_붙임.docx"
    # new_contract(a, b, c)