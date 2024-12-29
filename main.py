import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
airflow_home = os.getenv("AIRFLOW_HOME")

client = Groq(api_key=api_key)

def assemble_content(multiple_ai_models, eng):
    # Step 1: Product description

    product_description = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": f"Describe {multiple_ai_models}"}],
        temperature=0.7,
        max_tokens=300,
    )["choices"][0]["message"]["content"]

    # Step 2: Add SEO keywords
    seo_keywords = client.chat.completions.create(
        model="llama3-seo-8b-8192",
        messages=[{"role": "user", "content": product_description}],
        temperature=0,
        max_tokens=100,
    )["choices"][0]["message"]["content"]

    # Step 3: Grammar and style check
    corrected_text = client.chat.completions.create(
        model="llama3-grammar-corrector",
        messages=[{"role": "user", "content": product_description + " " + seo_keywords}],
        temperature=0,
        max_tokens=300,
    )["choices"][0]["message"]["content"]

    # Step 4: Translate into different languages
    translated_text = client.translate.create(
        text=corrected_text,
        model="llama3-translation-8b-8192",
        language_code=eng,
    )["translations"][0]["translatedText"]

    # Step 5: Sentiment analysis on customer feedback
    analysis = client.chat.completions.create(
        model="llama3-sentiment-analyzer",
        messages=[{"role": "user", "content": translated_text}],
        temperature=0,
        max_tokens=50,
    )["choices"][0]["message"]["content"]

    return analysis

product_name = "Apple iPhone"
language = "en"  # English
sentiment_analysis = assemble_content(product_name, language)
print(sentiment_analysis)