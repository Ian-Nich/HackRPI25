import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_script(prompt_text):
    """
    Generate a DJ script using GPT-4o-mini
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # free-tier model
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        max_tokens=250
    )

    return response.choices[0].message.content.strip()
