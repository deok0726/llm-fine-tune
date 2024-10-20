import os, torch, sys
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


def score_call(path):
    with open("/output/true.txt", 'w') as f:
        file_list = os.listdir(path)
        sorted_list = sorted(file_list)
        # sorted_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
        for text in sorted_list:
            print(text)
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)
            prompt = '''
                    카드사 고객 상담 내역 분석 담당 역할을 수행하세요. 당신의 업무는 카드사의 상담원과 고객 간 통화 내역을 분석하여 지정된 일곱 가지 기준에 따라 0점에서 3점까지 점수를 매기고, 이를 표로 정리해주는 것입니다. 
                    각 항목 별 점수에 대한 자세한 설명도 포함해 주세요. 표 아래에는 총점을 표시하고 점수 기준에 따른 민원 가능성을 알려주세요. 하나 이상의 통화 내역이 입력되는 경우는 동일한 고객과의 통화 내역이므로, 입력된 전체 통화 내역을 분석하여 하나의 결과로 반환해주세요. 
                    통화 시간이 여러 번 나뉜 경우, 각 통화의 마지막 줄에 hh:mm:ss 형색으로 기재된 ‘문장종료시간’을 합산하여 총 통화 시간을 계산하고, 이 값에 따라 점수를 부여하세요. 총 통화 시간이 10분을 초과할 경우 10분 단위로 추가 점수를 부여하세요. (자세한 내용은 기준 7번 항목을 참고하세요.) 통화 내역은 출력하지 마세요.

                    다음은 통화 내역 분석에 사용할 일곱 가지 기준입니다:
        
                    1. 금융감독원 및 민원 언급 (Mentioning 금융감독원 and Complaint)
                    - 고객이 대화 중 금융감독원이나 민원을 언급하였는지 여부. 금감원은 금융감독원의 줄임말임.
                    - 2점: 금융감독원이나 민원을 언급함
                    - 0점: 금융감독원이나 민원을 언급하지 않음
                    
                    2. 고객의 내부 정책 및 근거 요구 (Requesting Internal Policies and Supporting Reference)
                    - 고객이 대화 중 카드사의 내부 정책이나 근거를 직접적으로 요구했는지 여부. 카드사의 서비스 내용에 대한 단순 질의는 제외하고 서비스의 기준이 되는 내부 정책이나 근거 약관을 의미함. 해당 내용의 화자가 상담사인 경우는 포함하지 않음.
                    - 2점: 내부 정책 및 근거 직접적으로 언급
                    - 0점: 내부 정책 및 근거 직접적으로 언급하지 않음
                    
                    3. 고객의 불만 제기 빈도 (Frequency of Complaints)
                    - 고객이 대화 중 불만을 제기한 횟수. 횟수가 많을수록 높은 점수 부여. 고객의 단순 질의는 포함하지 않음.
                    - 1점: 불만 제기 없음
                    - 2점: 1~2회 불만 제기
                    - 3점: 3회 이상 불만 제기
                    
                    4. 상담원의 응대 태도 (Response Attitude)
                    - 상담원의 응대가 고객을 이해하고 문제를 해결하려는 태도를 보였는지 여부. 상담원의 응대가 고객을 진정시키거나 불만을 해결하지 못할수록 높은 점수 부여.
                    - 1점: 우수한 응대
                    - 2점: 보통 응대
                    - 3점: 미흡한 응대
                    
                    5. 고객의 언어 강도 (Intensity of Language)
                    - 고객이 사용하는 언어의 강도와 감정적 표현(예: 분노, 실망 등). 감정적 강도가 높을수록 높은 점수 부여.
                    - 1점: 감정적 표현 없음
                    - 2점: 보통 수준의 감정적 표현
                    - 3점: 높은 감정적 표현
                    
                    6. 문제 해결 여부 (Issue Resolution Status)
                    - 통화 중 고객의 문제 또는 불만이 해결되었는지 여부. 문제가 해결되지 않을수록 높은 점수 부여.
                    - 1점: 문제 완벽히 해결
                    - 2점: 문제 일부 해결
                    - 3점: 문제 해결되지 않음
                    
                    7. 통화 지속 시간 (Call Duration)
                    - 여러 통화의 '문장종료시간'을 합산한 총 통화 시간. 각 Sheet 마지막 줄의 ‘문장종료시간’을 기준으로 점수 부여. 점수 부여 시 실수하지 마세요. 정확도가 중요합니다.
                    - 1점: 총 통화 시간 0분 이상 10분 미만
                    - 2점: 총 통화 시간 10분 이상 20분 미만
                    - 3점: 총 통화 시간 20분 이상
            '''
            instruction = f'''
                        입력으로 받은 통화 내역 파일을 prompt의 일곱 가지 기준으로 평가하여 각 항목 별 점수와 근거, 항목 별 점수의 총점을 구해줘. 

                        통화 내역: {document_text}
                        항목 별 점수 및 근거:
                        총점:
                         '''

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
            
            # f.write(text + "\n")
            f.write(summary + "\n\n")
            f.write("-"*100 + "\n\n")
            torch.cuda.empty_cache()



if __name__ == "__main__":
    file_path = "/data/true"
    
    score_call(file_path)