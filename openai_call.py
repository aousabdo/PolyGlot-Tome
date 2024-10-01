from openai import OpenAI
import os

def openai_translate(text: str, source_lang: str = 'Arabic', target_lang: str = 'English'):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are an expert translator specializing in "
                                          f"{source_lang} to {target_lang} translations. "
                                          f"Your task is to provide accurate, nuanced, and "
                                          f"contextually appropriate translations while "
                                          f"preserving the original meaning, tone, and style "
                                          f"of the text."},
            {"role": "user", "content": f"Translate the following {source_lang} text to "
                                        f"{target_lang}. Ensure that you maintain the "
                                        f"original meaning, context, and nuances. If there "
                                        f"are any idioms, cultural references, or ambiguous "
                                        f"terms, provide the most appropriate translation "
                                        f"while preserving the intended meaning:\n\n{text}\n\n"
                                        f"Translation:"}
        ]
    )

    return completion.choices[0].message.content

# Example usage
if __name__ == "__main__":
    arabic_text = """
    مرحبا بكم في عالم الترجمة. الترجمة هي فن نقل المعاني والأفكار من لغة إلى أخرى. 
    إنها تتطلب مهارة وفهماً عميقاً للغتين المصدر والهدف، بالإضافة إلى معرفة بالثقافات المرتبطة بهما. 
    الترجمة الجيدة تحافظ على روح النص الأصلي مع جعله مفهوماً ومناسباً للجمهور المستهدف.
    """
    translated_text = openai_translate(arabic_text)
    print(translated_text)