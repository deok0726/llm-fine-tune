import transformers
import torch, os

model_id = "Bllossom/llama-3-Korean-Bllossom-70B"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
pipeline.model.eval()

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                full_text.append(stripped_line)
    return "\n".join(full_text)

def summarize_promotion(path):
    with open("/svc/project/genaipilot/fine-tune/output/result_oct_70b.txt", 'w') as f:
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

            prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                prompt,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            # print(outputs[0]["generated_text"][len(prompt):])
            result = outputs[0]["generated_text"][len(prompt):]
            
            print("LLM 답변: \n", result)
            f.write(result + "\n\n")
            f.write("-"*100 + "\n\n")


if __name__ == "__main__":
    # file_path = "01_업무위탁계약서(표준)2024 수정_비식별화 1.docx"
    file_path = "/svc/project/genaipilot/fine-tune/data/oct/"
    text = '/svc/project/genaipilot/fine-tune/output/result_oct.txt'

    text_a = '/svc/project/genaipilot/fine-tune/output/result_sep.txt'
    text_b = '/svc/project/genaipilot/fine-tune/output/result_oct.txt'

    summarize_promotion(file_path)
    # insight_promotion(text)
    # insight_compare(text_a, text_b)