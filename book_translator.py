import os
import argparse
import logging
import openai
import asyncio
import aiohttp
from typing import List
import re
import time
from tqdm.asyncio import tqdm_asyncio
import json
import google.generativeai as genai
from asyncio import Semaphore
from collections import defaultdict
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from environment variables
def load_config():
    return {
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', ''),
        'GOOGLE_GEMINI_API_KEY': os.environ.get('GOOGLE_GEMINI_API_KEY', ''),
        'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY', '')
    }

config = load_config()

# Set up API keys from configuration
OPENAI_API_KEY = config.get('OPENAI_API_KEY')
GOOGLE_GEMINI_API_KEY = config.get('GOOGLE_GEMINI_API_KEY')
ANTHROPIC_API_KEY = config.get('ANTHROPIC_API_KEY')

# Configure Gemini API
if GOOGLE_GEMINI_API_KEY:
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro-exp-0827")

# Validate API keys
if not OPENAI_API_KEY:
    logging.error("OpenAI API key is missing.")
if not GOOGLE_GEMINI_API_KEY:
    logging.warning("Google Gemini API key is missing.")
if not ANTHROPIC_API_KEY:
    logging.warning("Claude Sonnet API key is missing.")

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

def is_text_readable(text: str, threshold: float = 0.7) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > threshold

def split_text_into_sentences(text: str) -> List[str]:
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return sentences

def chunk_text(sentences: List[str], max_tokens: int, overlap: int = 1) -> List[str]:
    chunks = []
    current_chunk = []
    current_length = 0

    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence_length = len(sentence.split())

        if current_length + sentence_length > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = sentences[max(0, i - overlap):i + 1]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Rate limiter class
class RateLimiter:
    def __init__(self, max_calls, period):
        self.semaphore = Semaphore(max_calls)
        self.period = period
        self.last_reset = time.time()
        self.call_times = []

    async def acquire(self):
        now = time.time()
        if now - self.last_reset > self.period:
            self.last_reset = now
            self.call_times = []
            self.semaphore = Semaphore(self.semaphore._value)

        await self.semaphore.acquire()
        self.call_times.append(now)

    def release(self):
        self.semaphore.release()

# Dictionary to store rate limiters for each model
rate_limiters = defaultdict(lambda: RateLimiter(max_calls=2, period=60))

# Function to clean translation output
def clean_translation_output(output: str) -> str:
    patterns_to_remove = [
        r"^.*?Translation:\s*",
        r"^.*?The translation is:\s*",
        r"^.*?Translated text:\s*",
        r"^.*?Here is the translation:\s*",
        r"Please provide more context.*",
        r"If you provide more context.*",
        r"^.*?Explanation:\s*",
        r"^.*?Note:\s*",
    ]
    for pattern in patterns_to_remove:
        output = re.sub(pattern, '', output, flags=re.IGNORECASE | re.DOTALL)
    return output.strip()

# Modify the translate_chunk function
async def translate_chunk(session, chunk: str, source_lang: str, target_lang: str, model: str, retries: int = 3) -> str:
    logging.info(f"Translating chunk: {chunk[:100]}...")  # Log first 100 characters of the chunk

    rate_limiter = rate_limiters[model]
    await rate_limiter.acquire()

    try:
        for attempt in range(retries):
            try:
                if model == 'google_gemini':
                    logging.info(f"Translating chunk with Google Gemini.")
                    translated_chunk = await google_gemini_translate(session, chunk, source_lang, target_lang)
                elif model == 'openai':
                    logging.info(f"Translating chunk with OpenAI.")
                    translated_chunk = await openai_translate(chunk, source_lang, target_lang)
                elif model == 'claude_sonnet':
                    translated_chunk = await claude_sonnet_translate(session, chunk, source_lang, target_lang)
                else:
                    raise ValueError('Invalid model selected.')

                logging.info(f"Translation result (attempt {attempt + 1}): {translated_chunk[:100]}...")
                if translated_chunk.strip():
                    return translated_chunk
                else:
                    logging.warning(f"Empty translation result on attempt {attempt + 1}. Retrying...")
            except Exception as e:
                logging.error(f"Error during translation (attempt {attempt + 1}/{retries}): {e}")

            if attempt < retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

        logging.error("Maximum retries reached. Returning empty string.")
        return ''
    finally:
        rate_limiter.release()

async def google_gemini_translate(session, text: str, source_lang: str, target_lang: str) -> str:
    prompt = f"""
Please translate the following text from {source_lang} to {target_lang}:

{text}

Provide only the translation in {target_lang} without any explanations or additional text.
"""
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=4000,
                temperature=0,
                top_p=1.0,
                top_k=0,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
            }
        )

        if response.text:
            translated_text = response.text.strip()
            translated_text = clean_translation_output(translated_text)
            logging.info(f"Raw translation result: {translated_text[:100]}...")
            return translated_text
        else:
            logging.error("Empty response from Gemini API")
            return ""
    except Exception as e:
        logging.error(f"Google Gemini API request failed: {str(e)}")
        raise

async def openai_translate(text: str, source_lang: str = 'Arabic', target_lang: str = 'English'):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are an expert translator specializing in translating text from {source_lang} to {target_lang}. You provide only the translation without any explanations, notes, or additional content."},
                {"role": "user", "content": f"Please translate the following text from {source_lang} to {target_lang}:\n\n{text}\n\nProvide only the translation in {target_lang} and nothing else."}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API request failed: {str(e)}")
        raise

async def claude_sonnet_translate(session, text: str, source_lang: str, target_lang: str) -> str:
    # Implement the actual asynchronous API call to Claude Sonnet here
    # Placeholder implementation
    return text  # Replace with actual translation

def save_as_text(translated_text: str, output_path: str):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
    except Exception as e:
        logging.error(f"Failed to save text file: {e}")
        raise

def save_as_word(translated_text: str, output_path: str):
    try:
        from docx import Document
        document = Document()
        for paragraph in translated_text.split('\n'):
            document.add_paragraph(paragraph)
        document.save(output_path)
    except Exception as e:
        logging.error(f"Failed to save Word document: {e}")
        raise

def save_as_pdf(translated_text: str, output_path: str):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io

        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        textobject = can.beginText()
        textobject.setTextOrigin(50, 750)
        for line in translated_text.split('\n'):
            textobject.textLine(line)
        can.drawText(textobject)
        can.save()
        with open(output_path, 'wb') as f:
            f.write(packet.getvalue())
    except Exception as e:
        logging.error(f"Failed to save PDF file: {e}")
        raise

# Modify the main function to use a smaller batch size and add a delay between batches
async def main():
    parser = argparse.ArgumentParser(description='Automate the translation of books from one language to another.')
    parser.add_argument('pdf_path', help='Path to the input PDF file.')
    parser.add_argument('--source_lang', default='Arabic', help='Source language (default: Arabic).')
    parser.add_argument('--target_lang', default='English', help='Target language (default: English).')
    parser.add_argument('--model', choices=['google_gemini', 'openai', 'claude_sonnet'], default='google_gemini', help='LLM model to use for translation.')
    parser.add_argument('--output_format', choices=['text', 'word', 'pdf'], default='text', help='Output format of the translated text.')
    parser.add_argument('--output_path', help='Path to save the translated output.')
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens per chunk.')
    parser.add_argument('--overlap', type=int, default=2, help='Number of sentences to overlap between chunks.')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for failed API calls.')
    args = parser.parse_args()

    logging.info(f"Current working directory: {os.getcwd()}")
    full_path = os.path.abspath(args.pdf_path)
    logging.info(f"Attempting to access file: {full_path}")

    if not os.path.exists(full_path) or not os.access(full_path, os.R_OK):
        logging.error(f"Cannot access the file {full_path}.")
        return

    # Load and extract text from PDF
    logging.info("Extracting text from PDF...")
    try:
        # Implement your PDF text extraction logic here
        # For placeholder, we'll read text from a file
        with open(full_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        sentences = split_text_into_sentences(full_text)
        chunks = chunk_text(sentences, args.max_tokens - 500 if args.max_tokens else 1548, args.overlap)

        translated_text = ""

        async with aiohttp.ClientSession() as session:
            tasks = [translate_chunk(session, chunk, args.source_lang, args.target_lang, args.model, args.retries) for chunk in chunks]
            translated_chunks = await tqdm_asyncio.gather(*tasks, desc="Translating")

        translated_text = '\n'.join(translated_chunks)

        if not args.output_path:
            base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
            extension = 'txt' if args.output_format == 'text' else ('docx' if args.output_format == 'word' else 'pdf')
            args.output_path = f'{base_name}_translated.{extension}'

        logging.info(f"Saving translated text as {args.output_format.upper()}...")
        try:
            if args.output_format == 'text':
                save_as_text(translated_text, args.output_path)
            elif args.output_format == 'word':
                save_as_word(translated_text, args.output_path)
            elif args.output_format == 'pdf':
                save_as_pdf(translated_text, args.output_path)
        except Exception as e:
            logging.error(f"Failed to save output file: {e}")
            return

        logging.info(f'Translation completed. Output saved to {args.output_path}')

    except Exception as e:
        logging.error(f"Failed to process the PDF: {e}")
        return

if __name__ == '__main__':
    asyncio.run(main())
