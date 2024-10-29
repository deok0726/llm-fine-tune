import os, sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

dotenv_path = os.path.join(os.path.dirname(__file__), 'config/.env')
load_dotenv(dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")


def read_text_file(file_path):
    title = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = [title]
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                full_text.append(stripped_line)
    return "\n".join(full_text)

def generate_natural_language_answer(llm, document_text):
    answer_prompt_template = '''
        카드사의 상담원과 고객 간 통화 내역을 분석하여 고객이 통화 후 금융감독원에 민원을 제기할 가능성이 높은 통화 내역을 찾아야 합니다. 
        입력 파일들은 실제 상담원과 고객간 통화 내역으로, 이후 금융감독원 민원으로 접수된 통화 내역입니다. 'Sheet: 숫자'는 한 고객과 이루어진 통화의 횟수를 의미합니다.
        통화 내역들을 기반으로 금융감독원 민원 전이 가능성을 판단할 수 있는 핵심 지표들을 뽑아주세요.

        통화 내역 모음: {document_text}
        특징: '''
    
    answer_prompt = answer_prompt_template.format(document_text=document_text)
    response = llm.invoke(answer_prompt)
    
    return response.content

def criterion_call(path):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=None, openai_api_key=openai_api_key)

    file_list = os.listdir(path)
    sorted_list = sorted(file_list)
    document_text = ""
    for text in sorted_list:
        file_path = os.path.join(path, text)
        document_text += read_text_file(file_path) + "\n\n"

    result = generate_natural_language_answer(llm, document_text)
    print(result)

if __name__ == "__main__":
    criterion_call("/svc/project/genaipilot/fss_predict/del")