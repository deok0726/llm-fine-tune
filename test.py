import PyPDF2
import requests
from io import BytesIO
import os

def pdf_to_text(pdf_content):
    pdf_reader = PyPDF2.PdfReader(pdf_content)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def process_pdf_links(pdf_links,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, pdf_link in enumerate(pdf_links):
        response = requests.get(pdf_link)
        if response.status_code == 200:
            pdf_content = BytesIO(response.content)
            text = pdf_to_text(pdf_content)
            txt_filename = f"Research_Paper{idx+1}.txt"
            txt_path = os.path.join(output_folder, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            print(f"Text extracted from PDF {idx+1}")

# Example usage:
pdf_links = [
    'https://arxiv.org/pdf/2307.06090',
    # Add more PDF links as needed
]
output_folder = './raw_txt_input/'
process_pdf_links(pdf_links,output_folder)