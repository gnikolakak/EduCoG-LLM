from openai import OpenAI  

# Creates an instance of the OpenAI client using your API key
client = OpenAI(
    api_key="enter your API key"
)

try: # Tries to open the file prompt.txt in read mode with UTF-8 encoding
    with open("prompt.txt", "r", encoding="utf-8") as prompt_file:  
        prompt = prompt_file.read().strip() 
except FileNotFoundError: 
    print("Error: The file 'prompt.txt' was not found. Please check the current working directory and the file name.")
    exit(1)

# Sends the prompt to the OpenAI API using the gpt-4o-mini model
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True, 
    messages=[ 
        {"role": "user", "content": prompt}
    ]
)

output_message = completion.choices[0].message.content 

with open("output.txt", "w", encoding="utf-8") as output_file:
    output_file.write(output_message) 
    
print("Output has been written to output.txt") 


