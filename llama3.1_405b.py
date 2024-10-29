import os, sys, time
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv
from httpcore import RemoteProtocolError

dotenv_path = os.path.join(os.path.dirname(__file__), 'config/.env')
load_dotenv(dotenv_path)
api_key = os.getenv("NVIDIA_API_KEY")


def read_text_file(file_path):
    title = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = [title]
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                full_text.append(stripped_line)
    return "\n".join(full_text)


def read_word_file(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            full_text.append(paragraph.text)
    return "\n".join(full_text)


def summary_contract(path):
    document_text = read_word_file(path)
    
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )
    
    system_instruction = '''당신의 역할은 계약서의 핵심 내용을 파악하고, 이를 요약하는 것입니다. 아래 지시사항을 따라서 계약서 요약을 수행합니다.'''
    user_prompt = f'''# 지시사항
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
    
    completion = client.chat.completions.create(
    model="meta/llama-3.1-405b-instruct",
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.1,
    top_p=0.5,
    max_tokens=1024,
    stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

def compare_contract(a, b, c):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

    with open("/svc/project/llm-fine-tune/compare_contract.txt", 'w') as f:
        document_text_a = read_word_file(a)
        document_text_b = read_word_file(b)
        document_text_c = read_word_file(c)
        system_instruction = '''업로드된 문서들은 특정 계약 내용을 포함한 계약서입니다.
                            당신의 역할은 각 계약서의 내용을 분석하고, 롯데카드와 A사 간 계약 내용을 롯데카드와 다른 회사들(B사, C사) 간 계약 내용과 비교하는 것입니다.
                            계약서의 종류는 크게 2가지로 구분됩니다.
                            - 표준위탁계약서 : 롯데카드와 다른 회사의 계약 상 관계를 판단할 수 있는 계약서
                            - 각 붙임 계약서 : 롯데카드와 A사/B사/C사 간 계약서 내용 차이를 파악하기 위한 계약서
                            '''
        user_prompt = f'''
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

                        답변:'''
        
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.5,
            max_tokens=1024,
            stream=True
            )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                f.write(chunk.choices[0].delta.content)

def renew_contract(old, new):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

    with open("/svc/project/llm-fine-tune/renew_1.txt", 'w') as f:
        old = read_word_file(old)
        new = read_word_file(new)
        system_instruction = '''두 문서는 회사간 계약 내용에 대한 내용을 담고 있습니다. 
                                ‘22년 계약서'는 '24년 계약서'의 갱신 계약입니다.
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
                            '''
        user_prompt = f'''
                             ### 아래 입력 파일을 기반으로 양식에 맞게 답해주세요.

                                22년 계약서: {old}
                                24년 계약서: {new}
                                
                                답변: '''
        
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.5,
            max_tokens=2048,
            stream=True
            )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                f.write(chunk.choices[0].delta.content)

def criterion_call(path):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

    file_list = os.listdir(path)
    sorted_list = sorted(file_list)
    document_text = ""
    for text in sorted_list:
        file_path = os.path.join(path, text)
        document_text += read_text_file(file_path) + "\n\n"

    # print(document_text)
    # print("-"*100)

    system_instruction = '''
                    카드사의 상담원과 고객 간 통화 내역을 분석하여 고객이 통화 후 금융감독원에 민원을 제기할 가능성이 높은 통화 내역을 찾아야 합니다. 
                    입력 파일들은 실제 상담원과 고객간 통화 내역으로, 이후 금융감독원 민원으로 접수된 통화 내역입니다. 'Sheet: 숫자'는 한 고객과 이루어진 통화의 횟수를 의미합니다.
                    통화 내역들을 기반으로 금융감독원 민원 전이 가능성을 판단할 수 있는 핵심 지표들을 뽑아주세요.'''
    
    user_prompt = f'''
                    통화 내역 모음: {document_text}
                    특징: '''

    completion = client.chat.completions.create(
    model="meta/llama-3.1-405b-instruct",
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        top_p=0.5,
        max_tokens=1024,
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

def customer_call(path):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

    with open("/svc/project/llm-fine-tune/false.txt", 'w') as f:
        file_list = os.listdir(path)
        sorted_list = sorted(file_list)
        for text in sorted_list:
            print(text)
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)
        
            system_instruction = '''카드사 고객 상담 내역 분석 담당 역할을 수행하세요. 당신의 업무는 카드사의 상담원과 고객 간 통화 내역을 분석하여 지정된 일곱 가지 기준에 따라 0점에서 3점까지 점수를 매기고, 이를 표로 정리해주는 것입니다.
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
                            - 3점: 총 통화 시간 20분 이상'''
            user_prompt = f'''입력으로 받은 통화 내역 파일을 prompt의 일곱 가지 기준으로 평가하여 각 항목 별 점수와 근거, 항목 별 점수의 총점을 구해줘.

                                통화 내역: {document_text}
                                항목 별 점수 및 근거:
                                총점:'''

            completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.5,
            max_tokens=1024,
            stream=True
            )

            f.write(text + "\n")
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
                    f.write(chunk.choices[0].delta.content)
            f.write("\n" + "-"*100 + "\n")

def summary_promotion(path, month):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

    with open(f"/svc/project/llm-fine-tune/summary_{month}_405b.txt", 'w') as f:
        for text in os.listdir(path):
            print("\n[", text, "]", "\n")
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)
            system_instruction = '''당신은 카드사의 캐시백 혜택 요약을 수행합니다. 카드사의 특정 캐시백 프로모션에 대한 세부 정보를 제공하면 즉시 확인할 수 있는 요약 정보로 압축해주세요.'''
            user_prompt = f''' # 지시사항 1
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

            completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.5,
            max_tokens=1024,
            stream=True
            )

            f.write(text + "\n")
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
                    f.write(chunk.choices[0].delta.content)
            print("\n" + "-"*200 + "\n")
            f.write("\n" + "-"*100 + "\n")

def summary_promotion_retry(path, month):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=300
    )

    MAX_RETRIES = 5
    RETRY_DELAY = 5

    print(f"{month} 프로모션")
    with open(f"/svc/project/llm-fine-tune/summary_{month}_405b.txt", 'w') as f:
        for text in os.listdir(path):
            print("\n[", text, "]", "\n")
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)

            system_instruction = '''당신은 카드사의 캐시백 혜택 요약을 수행합니다. 카드사의 특정 캐시백 프로모션에 대한 세부 정보를 제공하면 즉시 확인할 수 있는 요약 정보로 압축해주세요.'''
            user_prompt = f''' # 지시사항 1
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

            for attempt in range(MAX_RETRIES):
                try:
                    completion = client.chat.completions.create(
                        model="meta/llama-3.1-405b-instruct",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        top_p=0.5,
                        max_tokens=1024,
                        stream=True
                    )

                    f.write(text + "\n")
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            print(chunk.choices[0].delta.content, end="")
                            f.write(chunk.choices[0].delta.content)

                    print("\n" + "-"*200 + "\n")
                    f.write("\n" + "-"*100 + "\n")
                    break

                except RemoteProtocolError as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print(f"Retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print("Max retries reached. Moving to the next file.")

def insight_promotion(path, month):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

    with open(f"/svc/project/llm-fine-tune/insight_promotion_{month}_405b.txt", 'w') as f:
        document_text = read_text_file(path)
        system_instruction = '''당신은 카드사의 캐시백 혜택 요약으로부터 질문에 해당하는 답변을 제공합니다. 요약 내용에 기반하여 시사점을 도출할 수 있는 답변을 생성해주세요.'''
        user_prompt = f''' 이 문서의 주요 내용은 10월 뱅크샐러드에서 진행하는 카드사 프로모션 요약입니다.
                    문서의 내용을 살펴보고, 분석하여 다음 카테고리에 맞게 설명을 부탁드립니다. 

                    1) 가장 높은 금액의 혜택을 제공하는 카드사와 그 카드사가 제공하는 혜택 금액은 얼마인가요?
                    2) 연회비가 가장 낮은 카드를 제공하는 카드사는 어디인가요? 연회비가 없는 경우, 그 카드가 가장 낮은 연회비를 제공하는 카드입니다.
                    3) 카드사 별 프로모션 내용 중 트렌드로 꼽을만한 것이 있다면 어떤게 있을지 알려주세요.
                    4) 당신이 생각하는 최고의 프로모션은 무엇인지 고민해보고, 1개만 골라주세요. 그리고 선정한 이유에 대해서도 구체적으로 설명해주세요. 최고의 프로모션 선정할 때는 네가 뱅크샐러드 홈페이지에 들어가서, 여러 카드사의 프로모션 중 하나를 선택한다고 생각하고 골라주세요.. 최고 프로모션을 선정할 때는 다양한 할인 카테고리와 구체적인 할인율에 대해서 설명해주세요.
                    5) 연회비 대비 최고 혜택을 제공하는 카드사 프로모션을 골라보고, 그 이유를 설명해주세요.

                    
                    프로모션 요약 : {document_text}
                    답변 : '''

        completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        top_p=0.5,
        max_tokens=1024,
        stream=True
        )

        print("\n[", month, "프로모션 정리 ]", "\n")
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                f.write(chunk.choices[0].delta.content)
        print("\n" + "-"*200 + "\n")

def insight_compare(text_a, text_b):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

    document_a = read_text_file(text_a)
    document_b = read_text_file(text_b)

    with open("/svc/project/llm-fine-tune/compare_promotion.txt", 'w') as f:
        system_instruction = '''당신은 카드사의 캐시백 혜택 요약으로부터 질문에 해당하는 답변을 제공합니다. 요약 내용에 기반하여 시사점을 도출할 수 있는 답변을 생성해주세요.'''
        user_prompt = f''' 제공한 문서의 내용은 각각 9월, 10월에 뱅크샐러드에서 진행된 카드사별 프로모션 내용에 대한 요약입니다. 
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
                        답변 :  '''

        completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        top_p=0.5,
        max_tokens=1024,
        stream=True
        )

        print("\n[ 9월/10월 프로모션 비교 ]", "\n")
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                f.write(chunk.choices[0].delta.content)

def summarize_news(client, path):
    MAX_RETRIES = 5
    RETRY_DELAY = 5
    print("뉴스 요약")

    with open("/svc/project/llm-fine-tune/output/summary_news_405b.txt", 'w') as f:
        file_list = os.listdir(path)
        sorted_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
        for text in sorted_list:
            print("\n[", text, "]", "\n")
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)
            system_instruction = '''롯데카드는 대한민국의 신용카드 회사이며, 로카는 롯데카드의 줄임말이고, LOCA는 롯데카드의 약자입니다. 당신은 롯데카드의 온라인 뉴스 관리 직원입니다. 당신의 임무는 인터넷에서 뉴스 기사를 찾아 요약하고 분석하는 것입니다. 업로드되는 txt 파일에는 뉴스 기사의 링크, 제목, 출처, 날짜, 키워드, 내용이 포함되어 있습니다.'''
            user_prompt = f'''기사 요약 생성
                            제공된 파일에서 다음 내용을 순서대로 추출하세요. 각 카테고리별로 줄을 변경하세요:
                            - 제목: 제목은 txt 파일의 세 번째 줄입니다. 제목만 포함하세요.
                            - 날짜: 날짜는 txt 파일의 네 번째 줄입니다. yyyy-mm-dd 형식으로 작성합니다.
                            - 출처: 출처는 txt 파일의 일곱 번째 줄입니다.
                            - 키워드: 키워드는 txt 파일의 첫 번째 줄입니다. 키워드가 큰따옴표 사이에 있는 경우 큰따옴표 없이 추출합니다.
                            - 요약: 글의 내용을 글머리 기호 3개로 요약합니다. 특정 회사명, 특히 롯데카드가 언급된 경우 요약에 명시하세요. 숫자가 언급된 경우 숫자를 요약에 포함하세요. 글머리 기호에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요.
                            - 링크: 링크는 txt 파일의 두 번째 줄입니다.
                            명확한 목소리 톤으로 작성하세요. 한국어로 답변해 주세요. 요약은 문장의 끝이 ~음., ~함. 과 같이 끝나는 음슴체로 작성해주세요. 코드 블록을 사용하지 마세요.
                            차근차근 생각하세요.

                            입력 파일: {document_text}
                            답변:  '''
            for attempt in range(MAX_RETRIES):
                try:
                    completion = client.chat.completions.create(
                        model="meta/llama-3.1-405b-instruct",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        top_p=0.5,
                        max_tokens=4096,
                        stream=True
                    )

                    f.write(text + "\n")
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            print(chunk.choices[0].delta.content, end="")
                            f.write(chunk.choices[0].delta.content)
                    print("\n" + "-"*200 + "\n")
                    f.write("\n" + "-"*100 + "\n")
                    break
                
                except RemoteProtocolError as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print(f"Retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print("Max retries reached. Moving to the next file.")
            
def summarize_sns(client, path):
    MAX_RETRIES = 5
    RETRY_DELAY = 5
    print("SNS 요약")

    with open("/svc/project/llm-fine-tune/output/summary_sns_405b.txt", 'w') as f:
        file_list = os.listdir(path)
        sorted_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
        for text in sorted_list:
            print("\n[", text, "]", "\n")
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)
            system_instruction = '''롯데카드는 대한민국의 신용카드 회사이며, 로카는 롯데카드의 줄임말이고, LOCA는 롯데카드의 약자입니다. 당신은 신용카드 회사인 롯데카드의 평판 관리 직원입니다. 여러분의 임무는 롯데카드에 대한 뉴스와 소셜 미디어 평판을 수집하여 요약과 인사이트를 얻는 것입니다. 업로드되는 txt 파일에는 커뮤니티 게시물의 키워드, 링크, 제목, 작성일, 추천 수, 댓글 수, 출처, 본문, 반응이 포함되어 있습니다.'''
            user_prompt = f'''게시글 요약 생성
                            제공된 파일에서 다음 내용을 순서대로 추출해 주세요. 각 카테고리별로 줄을 변경하세요:
                            - 링크: 링크는 txt 파일의 두 번째 줄입니다.
                            - 제목: 제목은 txt 파일의 세 번째 줄입니다. 제목만 포함합니다.
                            - 작성 날짜: 날짜는 txt 파일의 네 번째 줄입니다. '##시간 전'이라고 표시된 경우 현재 날짜와 시간을 사용하여 글의 작성 날짜를 계산합니다. yyyy-mm-dd 형식으로 작성합니다.
                            - 출처: 출처는 txt 파일의 일곱 번째 줄입니다.
                            - 요약: 본문 내용 중 롯데카드와 관련된 내용을 요약합니다. 롯데카드 관련 내용을 불릿 2개로 요약해주세요. 롯데카드 관련 숫자가 언급된 경우 해당 숫자를 요약에 포함하세요. 글머리 기호에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요.
                            - 반응: 본문의 내용에 대해 댓글의 사람들이 어떻게 반응하는지 한 문장으로 요약해주세요. 댓글이 없는 경우 '댓글 없음'으로 작성합니다. 그런 다음 추천 수와 댓글 수를 “추천: #, 댓글: #”으로 작성합니다. 추천 수와 댓글 수는 txt 파일의 다섯 번째와 여섯 번째 줄입니다.
                            - 성향: 이 글의 내용과 댓글이 롯데카드에 대해 긍정적(긍정), 중립(중립), 부정(부정)인지 판단합니다. 긍정에는 칭찬/추천, 중립에는 정보/질문, 부정에는 비판/비추천이 포함됩니다. '긍정'/'중립'/'부정' 중 하나로 분류해 주세요.
                            명확한 어조로 답변해 주세요. 한국어로 답변해 주세요. 요약과 반응은 문장의 끝이 ~음., ~함. 과 같이 끝나는 음절로 작성해 주세요. 코드 블록을 사용하지 마세요.
                            차근차근 생각하세요.

                            입력 파일: {document_text}
                            답변: '''
            for attempt in range(MAX_RETRIES):
                try:
                    completion = client.chat.completions.create(
                        model="meta/llama-3.1-405b-instruct",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        top_p=0.5,
                        max_tokens=4096,
                        stream=True
                    )

                    f.write(text + "\n")
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            print(chunk.choices[0].delta.content, end="")
                            f.write(chunk.choices[0].delta.content)
                    print("\n" + "-"*200 + "\n")
                    f.write("\n" + "-"*100 + "\n")
                    break
                    
                except RemoteProtocolError as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print(f"Retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print("Max retries reached. Moving to the next file.")

def insight_news(client, text):
    with open("/svc/project/llm-fine-tune/output/insight_news_405b.txt", 'w') as f:
        print("뉴스 분석")
        document_text = read_text_file(text)
        system_instruction = '''롯데카드는 대한민국의 신용카드 회사이며, 로카는 롯데카드의 줄임말이고, LOCA는 롯데카드의 약자입니다. 여러분은 롯데카드의 온라인 뉴스 관리 직원입니다. 여러분의 임무는 인터넷에서 뉴스 기사를 찾아 요약하고 분석하는 것입니다. 업로드되는 txt 파일에는 뉴스 기사의 제목, 날짜, 출처, 키워드, 요약, 링크가 포함되어 있습니다.'''
        user_prompt = f'''
                        분석 인사이트 생성
                        파일에서 다음 인사이트를 추출합니다. 특정 회사명이 언급된 경우 인사이트에 명시합니다. 카테고리 인사이트 이름은 괄호를 사용하지 않고 한글로만 작성합니다. 글머리 기호를 사용하여 요약합니다. 표나 코드 블록을 사용하지 마세요. 정확성이 매우 중요합니다.
                        1. 콘텐츠 요약(기사 요약)
                        '검색 키워드'에 '신용카드', '카드론', '카드대출', '카드페이먼트'가 포함된 콘텐츠를 요약합니다. 모든 키워드를 포함해야 합니다. 위에 명시된 키워드가 포함된 기사가 없는 경우, 해당 키워드에 글머리 기호를 만들지 마세요. 글머리 기호를 사용하여 '검색 키워드'를 기준으로 뉴스 기사를 그룹으로 분류합니다. '제목', '요약'을 통해 각 키워드의 내용을 3~5개의 글머리 기호로 요약합니다. 특정 회사 이름이 언급된 경우 요약에 포함하되 괄호로 강조하지 마세요. 숫자가 언급된 경우 요약에 포함하세요. 정확성이 매우 중요합니다. 그룹에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요.
                        2. 금융시장 동향 (금융시장 동향)
                        '검색 키워드'에 '기준금리', '원/달러'가 포함된 내용을 요약합니다. 모든 키워드를 포함해야 합니다. 해당 키워드가 포함된 기사가 없는 경우, 해당 키워드에 글머리 기호를 만들지 않습니다. 글머리 기호를 사용하여 '검색 키워드'를 기준으로 뉴스 기사를 그룹으로 묶습니다. '제목', '요약'을 통해 각 키워드의 내용을 3~5개의 글머리 기호로 요약합니다. 특정 회사 이름이 언급된 경우 요약에 포함하되 괄호로 강조하지 마세요. 숫자가 언급된 경우 요약에 포함하세요. 정확성이 매우 중요합니다. 그룹에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요.
                        3. 금융기관 동향(금융기관 동향)
                        '검색 키워드'에 '금융감독원', '금융위원회'가 포함된 내용을 요약합니다. 반드시 모든 키워드를 포함해야 합니다. 해당 키워드가 포함된 기사가 없는 경우, 해당 키워드에 글머리 기호를 만들지 않습니다. 글머리 기호를 사용하여 '검색 키워드'를 기준으로 뉴스 기사를 그룹으로 묶습니다. '제목', '요약'을 통해 각 키워드의 내용을 3~5개의 글머리 기호로 요약합니다. 특정 회사 이름이 언급된 경우 요약에 포함하되 괄호로 강조하지 마세요. 숫자가 언급된 경우 요약에 포함하세요. 정확성이 매우 중요합니다. 그룹에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요.
                        4. 주주사 동향 (주주사 동향)
                        '검색 키워드'에 'MBK'가 포함된 콘텐츠를 요약합니다. 모든 키워드를 포함해야 합니다. 해당 키워드가 포함된 기사가 없는 경우, 해당 키워드에 글머리 기호를 작성하지 않습니다. 글머리 기호를 사용하여 뉴스 기사를 '검색 키워드'를 기준으로 그룹으로 분류합니다. '제목', '요약'을 통해 각 키워드의 내용을 3~5개의 글머리 기호로 요약합니다. 특정 회사 이름이 언급된 경우 요약에 포함하되 괄호로 강조하지 마세요. 숫자가 언급된 경우 요약에 포함하세요. 정확성이 매우 중요합니다. 그룹에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요.
                        
                        단계별로 차근차근 생각하세요. 

                        입력 파일: {document_text}
                        답변:  '''

        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.5,
            max_tokens=4096,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                f.write(chunk.choices[0].delta.content)
        print("\n" + "-"*200 + "\n")

def insight_sns(client, text):
    with open("/svc/project/llm-fine-tune/output/insight_sns_405b.txt", 'w') as f:
        print("SNS 분석")
        document_text = read_text_file(text)
        system_instruction = '''롯데카드는 대한민국의 신용카드 회사이며, 로카는 롯데카드의 줄임말이고, LOCA는 롯데카드의 약자입니다. 당신은 신용카드 회사인 롯데카드의 평판 관리 직원입니다. 여러분의 임무는 롯데카드에 대한 뉴스와 소셜 미디어 평판을 수집하여 요약과 인사이트를 얻는 것입니다. 업로드되는 txt 파일에는 커뮤니티 게시물의 링크, 제목, 작성 날짜, 출처, 요약, 반응, 성향이 포함되어 있습니다.'''
        user_prompt = f'''
                        시사점 도출
                        파일에서 다음과 같은 인사이트를 추출합니다.
                        - 부정적 게시글 전반적 요약:
                        - '게시글 요약' 시트의 '성향' 란에 '부정'이 있는 콘텐츠를 요약합니다. '제목', '요약'을 읽고 3~5개의 글머리 기호로 내용을 요약합니다. 특정 회사명이 언급된 경우 요약에 포함합니다. 숫자가 언급된 경우 숫자를 요약에 포함하세요. 정확성이 매우 중요합니다. 그룹에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요. 한글로 요약하세요.
                        - 댓글이 많은 부정적 게시글 요약:
                        - '게시글 요약' 시트의 '성향' 칸에 '부정'이 있는 콘텐츠를 요약합니다. ‘성향’이 ‘부정’인 게시글 중 댓글이 가장 많은 세 게시글을 요약합니다. '제목', '요약'을 읽고 글머리 기호 3개로 내용을 요약합니다. '요약'과 '반응'의 내용은 위의 요약보다 더 구체적으로 요약하세요. 특정 회사명이 언급된 경우 해당 회사명을 요약에 포함하세요. 숫자가 언급된 경우 숫자를 요약에 포함하세요. 정확성이 매우 중요합니다. '추천' 및 '댓글'의 개수를 기재하지 마세요. 그룹에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요. 한글로 요약하세요.
                        차근차근 생각하세요.

                        입력 파일: {document_text}
                        답변: '''
        
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.5,
            max_tokens=4096,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                f.write(chunk.choices[0].delta.content)
        print("\n" + "-"*200 + "\n")

def table_output(client):
    with open("/svc/project/llm-fine-tune/output/insight_sns_405b.txt", 'w') as f:
        system_instruction = '''당신은 입력된 질문에 대해 답변을 표로 정리하여 생성하는 AI agent입니다.'''
        user_prompt = f'''
                        질문: 한국의 4대 회계법인 삼일, 삼정, 한영, 안진에 대해 직원수, 매출, 향후 전망 등을 비교해줘.
                        답변: '''
        
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.5,
            max_tokens=4096,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                f.write(chunk.choices[0].delta.content)
        print("\n" + "-"*200 + "\n")


if __name__ == "__main__":
    # old = "/svc/project/llm-fine-tune/data/contract/(1) 임대차 계약서/① '22년 임대차계약서.docx"
    # new = "/svc/project/llm-fine-tune/data/contract/(1) 임대차 계약서/① '24년 임대차계약서.docx"

    # renew_contract(old, new)

    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = api_key
    )

    # summarize_news(client, "/svc/project/llm-fine-tune/data/news")
    # summarize_sns(client, "/svc/project/llm-fine-tune/data/sns")
    # insight_news(client, "/svc/project/llm-fine-tune/output/summary_news_405b.txt")
    # insight_sns(client, "/svc/project/llm-fine-tune/output/summary_sns_405b.txt")

    criterion_call("/svc/project/genaipilot/fss_predict/del")

    # summary_promotion_retry("/svc/project/llm-fine-tune/data/sep", "sep")
    # summary_promotion_retry("/svc/project/llm-fine-tune/data/oct", "oct")
    # insight_promotion("/svc/project/llm-fine-tune/summary_sep_405b.txt", "sep")
    # insight_promotion("/svc/project/llm-fine-tune/summary_oct_405b.txt", "oct")
    # insight_compare("/svc/project/llm-fine-tune/insight_promotion_sep_405b.txt", "/svc/project/llm-fine-tune/insight_promotion_oct_405b.txt")

    # table_output(client)

