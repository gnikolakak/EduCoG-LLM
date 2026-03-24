import PyPDF2  
from openai import OpenAI 

def extract_text_from_pdf(pdf_path): 
    with open(pdf_path, 'rb') as file: 
        reader = PyPDF2.PdfReader(file)  
        text = ""  
        for page in reader.pages:  
            page_text = page.extract_text() 
            if page_text:  
                text += page_text  
        return text  

pdf_path = "document.pdf" 
try:  
    pdf_text = extract_text_from_pdf(pdf_path)
except FileNotFoundError:   
    print(f"Error: PDF '{pdf_path}' is not found.")
    exit(1)

prompt = f"""
Write 10 multiple choice questions about Artificial Intelligence and steepest descent according to the text below. Each question should have 5 possible answers and only one of them should be correct. The audience answering the questions should be at a university level. The questions and answers should be in Greek. The difficulty level of each question is on a Likert scale of 1 to 5 (1 = very easy and 5 = very difficult). Of the 10 questions, one has a difficulty level of 1, two have a difficulty level of 2, three have a difficulty level of 3, two have a difficulty level of 4, and the rest have a difficulty level of 5.

Text:
{pdf_text[:3000]}
"""

client = OpenAI(
    api_key="Enter your API key"
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

output_message = completion.choices[0].message.content

with open("output.txt", "w", encoding="utf-8") as output_file:
    output_file.write(output_message)

print("The questions were saved to the file 'output.txt'")


