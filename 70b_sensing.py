import os
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# ollama_llm = Ollama(model="llama3.1:latest", temperature=0.1)
ollama_llm = Ollama(model="llama3.1:70b", temperature=0.1)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                full_text.append(stripped_line)
    return "\n".join(full_text)

# Define the prompt
def summarize_sns(path):
    with open("/svc/project/genaipilot/fine-tune/output/sns_70b.txt", 'w') as f:
        file_list = os.listdir(path)
        sorted_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
        # sorted_list = sorted(file_list)
        for text in sorted_list:
            print(text)
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)
            template = """
                    # 역할
                    롯데카드는 대한민국의 신용카드 회사이며, 로카는 롯데카드의 줄임말이고, LOCA는 롯데카드의 약자입니다. 당신은 신용카드 회사인 롯데카드의 평판 관리 직원입니다. 여러분의 임무는 롯데카드에 대한 뉴스와 소셜 미디어 평판을 수집하여 요약과 인사이트를 얻는 것입니다. 업로드되는 txt 파일에는 커뮤니티 게시물의 키워드, 링크, 제목, 작성일, 추천 수, 댓글 수, 출처, 본문, 반응이 포함되어 있습니다.
                    
                    # 지시사항
                    게시글 요약 생성
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
                    답변: """
            
            prompt = PromptTemplate(
                input_variables=["document_text"],
                template=template,
            )
            formatted_prompt = prompt.format(document_text=document_text)
            response = ollama_llm.invoke(formatted_prompt)

            print("LLM 답변: \n", response)
            f.write(text + "\n")
            f.write(response + "\n\n")
            f.write("-"*100 + "\n\n")

def summarize_news(path):
    with open("/svc/project/genaipilot/fine-tune/output/news_70b.txt", 'w') as f:
        file_list = os.listdir(path)
        sorted_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
        # sorted_list = sorted(file_list)
        for text in sorted_list:
            print(text)
            file_path = os.path.join(path, text)
            document_text = read_text_file(file_path)
            template = """
                    # 역할
                    롯데카드는 대한민국의 신용카드 회사이며, 로카는 롯데카드의 줄임말이고, LOCA는 롯데카드의 약자입니다. 당신은 롯데카드의 온라인 뉴스 관리 직원입니다. 당신의 임무는 인터넷에서 뉴스 기사를 찾아 요약하고 분석하는 것입니다. 업로드되는 txt 파일에는 뉴스 기사의 링크, 제목, 출처, 날짜, 키워드, 내용이 포함되어 있습니다.
                    
                    # 지시사항
                    기사 요약 생성
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
                    답변: """
            
            prompt = PromptTemplate(
                input_variables=["document_text"],
                template=template,
            )
            formatted_prompt = prompt.format(document_text=document_text)
            response = ollama_llm.invoke(formatted_prompt)

            print("LLM 답변: \n", response)
            f.write(text + "\n")
            f.write(response + "\n\n")
            f.write("-"*100 + "\n\n")

def insight_sns(path):
    with open("/svc/project/genaipilot/fine-tune/output/sns_70b_insight.txt", 'w') as f:
        document_text = read_text_file(path)
        template = """
                # 역할
                롯데카드는 대한민국의 신용카드 회사이며, 로카는 롯데카드의 줄임말이고, LOCA는 롯데카드의 약자입니다. 당신은 신용카드 회사인 롯데카드의 평판 관리 직원입니다. 여러분의 임무는 롯데카드에 대한 뉴스와 소셜 미디어 평판을 수집하여 요약과 인사이트를 얻는 것입니다. 업로드되는 txt 파일에는 커뮤니티 게시물의 링크, 제목, 작성 날짜, 출처, 요약, 반응, 성향이 포함되어 있습니다.
                
                # 지시사항
                시사점 도출
                파일에서 다음과 같은 인사이트를 추출합니다.
                - 부정적 게시글 전반적 요약:
                - '게시글 요약' 시트의 '성향' 란에 '부정'이 있는 콘텐츠를 요약합니다. '제목', '요약'을 읽고 3~5개의 글머리 기호로 내용을 요약합니다. 특정 회사명이 언급된 경우 요약에 포함합니다. 숫자가 언급된 경우 숫자를 요약에 포함하세요. 정확성이 매우 중요합니다. 그룹에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요. 한글로 요약하세요.
                - 댓글이 많은 부정적 게시글 요약:
                - '게시글 요약' 시트의 '성향' 칸에 '부정'이 있는 콘텐츠를 요약합니다. ‘성향’이 ‘부정’인 게시글 중 댓글이 가장 많은 세 게시글을 요약합니다. '제목', '요약'을 읽고 글머리 기호 3개로 내용을 요약합니다. '요약'과 '반응'의 내용은 위의 요약보다 더 구체적으로 요약하세요. 특정 회사명이 언급된 경우 해당 회사명을 요약에 포함하세요. 숫자가 언급된 경우 숫자를 요약에 포함하세요. 정확성이 매우 중요합니다. '추천' 및 '댓글'의 개수를 기재하지 마세요. 그룹에 번호를 매기지 마세요. 코드 블록을 사용하지 마세요. 한글로 요약하세요.
                차근차근 생각하세요.

                입력 파일: {document_text}
                답변: """
        
        prompt = PromptTemplate(
            input_variables=["document_text"],
            template=template,
        )
        formatted_prompt = prompt.format(document_text=document_text)
        response = ollama_llm.invoke(formatted_prompt)

        print("LLM 답변: \n", response)
        f.write(response + "\n\n")
        f.write("-"*100 + "\n\n")

def insight_news(path):
    with open("/svc/project/genaipilot/fine-tune/output/news_70b_insight.txt", 'w') as f:
        document_text = read_text_file(path)
        template = """
                # 역할
                롯데카드는 대한민국의 신용카드 회사이며, 로카는 롯데카드의 줄임말이고, LOCA는 롯데카드의 약자입니다. 여러분은 롯데카드의 온라인 뉴스 관리 직원입니다. 여러분의 임무는 인터넷에서 뉴스 기사를 찾아 요약하고 분석하는 것입니다. 업로드되는 txt 파일에는 뉴스 기사의 제목, 날짜, 출처, 키워드, 요약, 링크가 포함되어 있습니다.
                
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
                단계별로 생각하세요. 

                입력 파일: {document_text}
                답변: """
        
        prompt = PromptTemplate(
            input_variables=["document_text"],
            template=template,
        )
        formatted_prompt = prompt.format(document_text=document_text)
        response = ollama_llm.invoke(formatted_prompt)

        print("LLM 답변: \n", response)
        f.write(response + "\n\n")
        f.write("-"*100 + "\n\n")

if __name__ == "__main__":
    summarize_sns('/svc/project/genaipilot/fine-tune/data/sns')
    summarize_news('/svc/project/genaipilot/fine-tune/data/news')

    insight_sns('/svc/project/genaipilot/fine-tune/output/sns_70b.txt')
    insight_news('/svc/project/genaipilot/fine-tune/output/news_70b.txt')