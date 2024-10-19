import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                full_text.append(stripped_line)
    return "\n".join(full_text)

def summarize_contract(text):
    document_text = read_text_file(file_path)
    prompt = '''Your job is to identify the key points of the contract and summarize them. Follow the instructions below to summarize the contract.당신의 역할은 계약서의 핵심 내용을 파악하고, 이를 요약하는 것입니다. 아래 지시사항을 따라서 계약서 요약을 수행합니다.'''
    instruction = f'''업무위탁계약서 파일을 보고, 계약 내용 중 핵심 내용들을 요약해주세요. 
                    - '계약 날짜', '총 계약 기간', '계약 금액', '계약 대상 업무(위탁 업무)', '서비스 수준'은 핵심 내용이므로, 답변에 반드시 포함해주세요.
                    - '총 계약 기간'은 '계약 날짜'부터 시작해야 합니다. 총 계약 기간은 다음과 같은 형식으로 답변해주세요. (ex. 총 계약 기간: 2024.01.02~2025.01.01)
                    - '서비스 수준'은 서비스 별 상세내용과 제공횟수에 대한 내용이 포함되어야 합니다.
                    
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
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.5,
        repetition_penalty = 1.1
    )
    
    summary = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return summary

def summarize_promotion(path):
    with open("/svc/project/genaipilot/fine-tune/output/result_oct.txt", 'w') as f:
        for text in os.listdir(path):
            print(text)
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)
            prompt = '당신은 카드사의 캐시백 혜택 요약을 수행합니다. 카드사의 특정 캐시백 프로모션에 대한 세부 정보를 제공하면 즉시 확인할 수 있는 요약 정보로 압축해주세요.'
            instruction = f'''
                            # 지시사항 1
                            입력으로 들어온 카드 프로모션 텍스트 파일을 요약해주세요.

                            # 지시사항 2
                            요약은 다음 8가지 카테고리로 구성되어야 합니다.:
                            1) 카드사명: 프로모션을 제공하는 카드사의 이름입니다.

                            2) 최대 혜택 금액: 이 프로모션에서 고객이 당첨될 수 있는 최대 금액입니다. 프로모션 내에 여러 이벤트가 있는 경우 최대 혜택 금액은 모든 이벤트의 합이 되어야 합니다. 답변에는 최대 혜택 금액만 포함합니다.
                            (예시) 프로모션에 이벤트 1: 90,000원, 이벤트 2: 10,000원, 이벤트 3: 30,000원 있다면 해당 프로모션의 최대 혜택 금액은 130,000원

                            3) 이벤트 혜택 대상: 프로모션에 참여할 수 있는 고객입니다. 이벤트가 여러 개 있는 경우 각 이벤트의 참가 자격을 설명해주세요. 
                            (예시) 이벤트 1: 지난 6개월 동안 활동이 없는 고객, 이벤트 2: 신규 고객

                            4) 이벤트 대상 카드와 이벤트 대상 카드의 혜택: 프로모션 대상 카드를 나열하고, 카드 별 혜택에 대해서 설명해주세요. '이벤트 대상 카드'가 여러 장 있는 경우 모든 카드를 포함해야 합니다.
                            (예시) 디지로카 Las Vegas - 국내외 가맹점 최대 2%,무이자 할부 2~3개월, 디지로카 Auto - 학원 최대 7%, 무이자 할부 2-3개월

                            5) 이벤트 대상 카드의 연회비: 각 이벤트 대상 카드의 연회비를 나열합니다. 카드가 여러 개인 경우 각 카드의 연회비를 누락 없이 기재해야 합니다. “카드 종류에 따라 연회비가 다릅니다"라는 답변이 아니라 모든 카드의 연회비를 세분화하여 기재해야 합니다. 
                            (예시)
                            - Point Plan : 국내용 2만원 / 해외겸용 2만 3천원
                            - 처음 : 국내용 1만 5천원 / 해외겸용 1만 8천원
                            - 신한카드 Air One : 국내용 4만 9천원 / 해외겸용 5만 1천원
                            - 아시아나 신한카드 Air 1.5 : 국내용 4만 3천원 / 해외겸용 4만 5천원
                            - Deep Oil : 국내용 1만원 / 해외겸용 1만 3천원
                            - Mr.Life : 국내용 1만 5천원 / 해외겸용 1만 8천원
                            - 플리 : 국내용 1만 5천원 / 해외겸용 1만 8천원
                            - B.Big(삑) : 국내용 1만원/해외겸용 1만 3천원

                            6) 이벤트의 상세내용: 고객이 캐시백을 받기 위해 지출해야 하는 금액을 설명합니다. '이벤트의 상세 내용'을 요약할 때는 모든 혜택 정보를 누락 없이 정확하게 포함해야 합니다.
                            (예시) 이벤트 1: 20만원 이상 사용 시 10만원 캐시백, 이벤트 2: 해외 가맹점에서 사용 시 6만원 캐시백

                            7) 이벤트 기간: 프로모션의 유효 기간입니다. '이벤트'가 여러 개 있는 경우, 각 '이벤트'의 대상 기간을 누락 없이 기재해야 합니다. 
                            (예시) 이벤트 1: 2024.09.01 ~ 2024.09.30, 이벤트 2: 2024.09.01 ~ 2024.10.15

                            
                            프로모션 내용 : {document_text}
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
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.5,
                repetition_penalty = 1.1
            )
            
            summary = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

            print("LLM 답변: \n", summary)
            f.write(summary + "\n\n")
            f.write("-"*100 + "\n\n")
        
def insight_promotion(text):
    document_text = read_text_file(text)
    with open("/svc/project/genaipilot/fine-tune/output/insight_oct.txt", 'w') as f:
        prompt = '당신은 카드사의 캐시백 혜택 요약으로부터 질문에 해당하는 답변을 제공합니다. 요약 내용에 기반하여 시사점을 도출할 수 있는 답변을 생성해주세요'
        instruction = f'''
                        이 문서의 주요 내용은 10월 뱅크샐러드에서 진행하는 카드사 프로모션 요약입니다.
                        문서의 내용을 살펴보고, 분석하여 다음 카테고리에 맞게 설명을 부탁드립니다. 

                        1) 가장 높은 금액의 혜택을 제공하는 카드사와 그 카드사가 제공하는 혜택 금액은 얼마인가요?
                        2) 연회비가 가장 낮은 카드를 제공하는 카드사는 어디인가요? 연회비가 없는 경우, 그 카드가 가장 낮은 연회비를 제공하는 카드입니다.
                        3) 카드사 별 프로모션 내용 중 트렌드로 꼽을만한 것이 있다면 어떤게 있을지 알려주세요.
                        4) 당신이 생각하는 최고의 프로모션은 무엇인지 고민해보고, 1개만 골라주세요. 그리고 선정한 이유에 대해서도 구체적으로 설명해주세요. 최고의 프로모션 선정할 때는 네가 뱅크샐러드 홈페이지에 들어가서, 여러 카드사의 프로모션 중 하나를 선택한다고 생각하고 골라주세요.. 최고 프로모션을 선정할 때는 다양한 할인 카테고리와 구체적인 할인율에 대해서 설명해주세요.
                        5) 연회비 대비 최고 혜택을 제공하는 카드사 프로모션을 골라보고, 그 이유를 설명해주세요.

                        
                        프로모션 요약 : {document_text}
                        답변 : '''
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
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.5,
            repetition_penalty = 1.1
        )
        
        insight = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        print("LLM 답변: \n", insight)
        f.write(insight + "\n\n")
        f.write("-"*100 + "\n\n")

def insight_compare(text_a, text_b):
    document_a = read_text_file(text_a)
    document_b = read_text_file(text_b)
    with open("/svc/project/genaipilot/fine-tune/output/insight_compare.txt", 'w') as f:
        prompt = '당신은 카드사의 캐시백 혜택 요약으로부터 질문에 해당하는 답변을 제공합니다. 요약 내용에 기반하여 시사점을 도출할 수 있는 답변을 생성해주세요'
        instruction = f'''
                        제공한 문서의 내용은 각각 9월, 10월에 뱅크샐러드에서 진행된 카드사별 프로모션 내용에 대한 요약입니다. 
                        저는 카드사 별 프로모션 내용이 9월, 10월 간 어떤 변화가 생겼는지 알고 싶습니다. 
                        두 문서의 내용을 분석하여, 다음 내용들에 대해 답변해주세요.

                        1) 9월과 비교했을 때 10월에 새롭게 추가되거나, 빠진 카드사가 있는지? 
                        ## IF 추가된 카드사가 있다면 어떤 혜택을 강조하는 프로모션을 제공하는지 설명해주세요. 
                        ## IF 빠진 카드사가 있다면 왜 프로모션을 종료했는지 그 이유를 추론해보세요.

                        2) 9월 카드사들의 프로모션 내용 트렌드와 10월 카드사들의 프로모션 트렌드에 대해서 분석해서 각각 설명해준 다음, 두 트렌드 간의 차이점이 있다면 그 차이점에 대해서 구체적으로 설명해주세요.

                        3) 9월과 10월 모두 프로모션을 진행하고 있는 카드사들의 경우, 9월 프로모션과 10월 프로모션의 내용을 비교해보고, 9월 대비 10월 달라진 내용이 있다면, 달라진 부분을 정리해서 알기 쉽게 알려주세요. 달라진 부분을 분석해서 결론도 함께 내려주세요. 반드시 모든 카드사 프로모션을 포함해야 합니다.
                        ## IF 달라진 부분이 있다면, 내용이 달라진 이유를 추론해보세요.

                        
                        9월 프로모션 요약 : {document_a}
                        10월 프로모션 요약 : {document_b}
                        답변 : '''
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
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.5,
            repetition_penalty = 1.1
        )
        
        insight = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        print("LLM 답변: \n", insight)
        f.write(insight + "\n\n")
        f.write("-"*100 + "\n\n")


if __name__ == "__main__":
    # file_path = "01_업무위탁계약서(표준)2024 수정_비식별화 1.docx"
    file_path = "/svc/project/genaipilot/fine-tune/data/oct/"
    text = '/svc/project/genaipilot/fine-tune/output/result_oct.txt'

    text_a = '/svc/project/genaipilot/fine-tune/output/result_sep.txt'
    text_b = '/svc/project/genaipilot/fine-tune/output/result_oct.txt'

    # summarize_promotion(file_path)
    # insight_promotion(text)
    insight_compare(text_a, text_b)