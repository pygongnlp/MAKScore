import openai
import re

openai.api_key = <YOUR_OPENAI_KEY>

prompt = "The summary evaluation task refers to evaluate the summary based on the news article.\n" \
         "Your task is to list nineteen aspects that can be considered when measuring the relevance of summary.\n" \
         "Relevance: The rating measures how well the summary captures the key points of the article. " \
         "Consider whether all and only the important aspects are contained in the summary."

while True:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": prompt},
            ],
            temperature=0,
            n=1
        )
        break
    except Exception as e:
        print(e)


content = response["choices"][0]["message"]["content"].split('\n')

aspects = {}
for c in content:
    if not c: continue
    c = c.split(":")
    aspects[re.sub(r'^[\d\s\W]+', '', c[0])] = c[1].strip()

print(aspects)
