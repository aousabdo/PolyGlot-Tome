import os
import argparse
import logging
import openai
import requests
import asyncio
import aiohttp
from typing import List
from PyPDF2 import PdfReader
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import time
import re
from tqdm.asyncio import tqdm_asyncio
import json
import google.generativeai as genai
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document as PDF2TextDocument

from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Add these at the top of your file, after the other imports
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
genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro-exp-0827")

# Validate API keys
if not OPENAI_API_KEY:
    logging.error("OpenAI API key is missing.")
if not GOOGLE_GEMINI_API_KEY:
    logging.warning("Google Gemini API key is missing.")
if not ANTHROPIC_API_KEY:
    logging.warning("Claude Sonnet API key is missing.")

# Set up OpenAI API key
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def is_text_readable(text: str, threshold: float = 0.7) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > threshold

def ocr_pdf(pdf_path: str) -> str:
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image, lang='ara+eng')
        return text
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        raise

def preprocess_arabic_text(text: str) -> str:
    # Remove non-Arabic characters and normalize Arabic text
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    arabic_text = ' '.join(arabic_pattern.findall(text))
    return arabic_text

def ocr_pdf_with_arabic(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    full_text = ""
    for image in images:
        # Specify multiple languages, with Arabic as primary
        text = pytesseract.image_to_string(image, lang='ara+eng', config='--psm 6')
        full_text += preprocess_arabic_text(text) + "\n\n"
    return full_text


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    try:
        # Try using multilingual_pdf2text first
        pdf_document = PDF2TextDocument(
            document_path=pdf_path,
            language='ara'
        )
        pdf2text = PDF2Text(document=pdf_document)
        extracted_data = pdf2text.extract()
        
        full_text = ""
        for page in extracted_data:
            full_text += preprocess_arabic_text(page['text']) + "\f"  # Add form feed character between pages
        
        # If the extracted text is too short or contains many non-Arabic characters, fall back to OCR
        if len(full_text.strip()) < 100 or len(re.findall(r'[a-zA-Z]', full_text)) > len(full_text) * 0.5:
            logging.info("Extracted text may be incorrect. Falling back to OCR...")
            full_text = ocr_pdf_with_arabic(pdf_path)
        
        page_count = full_text.count('\f') + 1  # Count form feed characters and add 1
        logging.info(f"Extracted {page_count} pages from PDF")
        logging.info(f"Sample of extracted text: {full_text[:1000]}...")  # Log first 1000 characters
        return full_text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        raise

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

# Add this class for rate limiting
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

# Add this dictionary to store rate limiters for each model
rate_limiters = defaultdict(lambda: RateLimiter(max_calls=2, period=60))

# Modify the translate_chunk function
async def translate_chunk(session, chunk: str, source_lang: str, target_lang: str, model: str, retries: int = 3) -> str:
    logging.info(f"Translating chunk: {chunk[:100]}...")  # Log first 100 characters of the chunk

    rate_limiter = rate_limiters[model]
    await rate_limiter.acquire()

    try:
        for attempt in range(retries):
            try:
                if model == 'google_gemini':
                    print(f"Translating chunk with Google Gemini: {chunk[:100]}...")
                    translated_chunk = await google_gemini_translate(session, chunk, source_lang, target_lang)
                elif model == 'openai':
                    print(f"Translating chunk with OpenAI: {chunk[:100]}...")
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
    Translate the following {source_lang} text to {target_lang}. Do not add any explanations or notes, just provide the translation:

    {text}

    Translation:
    """
    try:
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            max_output_tokens=4000,
            temperature=0.2,
            top_p=0.95,
            top_k=40
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
        })
        
        if response.text:
            translated_text = response.text.strip()
            # Remove any potential prefixes like "Translation:" or "Here's the translation:"
            translated_text = re.sub(r'^(Translation:|Here\'s the translation:)\s*', '', translated_text, flags=re.IGNORECASE)
            logging.info(f"Raw translation result: {translated_text[:100]}...")  # Log the raw result
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
    except Exception as e:
        logging.error(f"OpenAI API request failed: {str(e)}")
        raise

async def claude_sonnet_translate(session, text: str, source_lang: str, target_lang: str) -> str:
    # Implement the actual asynchronous API call to Claude Sonnet 3.5 here
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
        document = Document()
        for paragraph in translated_text.split('\n'):
            document.add_paragraph(paragraph)
        document.save(output_path)
    except Exception as e:
        logging.error(f"Failed to save Word document: {e}")
        raise

def save_as_pdf(translated_text: str, output_path: str):
    try:
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
    parser.add_argument('--autonomous', action='store_true', help='Run in autonomous mode without manual confirmations.')
    args = parser.parse_args()

    logging.info(f"Current working directory: {os.getcwd()}")
    full_path = os.path.abspath(args.pdf_path)
    logging.info(f"Attempting to access file: {full_path}")

    if not os.path.exists(full_path) or not os.access(full_path, os.R_OK):
        logging.error(f"Cannot access the file {full_path}.")
        return

    logging.info("Extracting text from PDF...")
    try:
        full_text = extract_text_from_pdf(args.pdf_path)
        pages = full_text.split('\f')
        total_pages = len(pages)
        logging.info(f"Extracted {total_pages} pages from PDF")

        batch_size = 2
        translated_text = ""
        
        for i in range(0, total_pages, batch_size):
            batch_pages = pages[i:i+batch_size]
            batch_text = '\f'.join(batch_pages)
            
            logging.info(f"Processing batch {i//batch_size + 1} of {(total_pages-1)//batch_size + 1} (pages {i+1}-{min(i+batch_size, total_pages)})")

            sentences = split_text_into_sentences(batch_text)
            chunks = chunk_text(sentences, args.max_tokens - 500 if args.max_tokens else 1548, args.overlap)

            async with aiohttp.ClientSession() as session:
                tasks = [translate_chunk(session, chunk, args.source_lang, args.target_lang, args.model, args.retries) for chunk in chunks]
                batch_translated_chunks = await tqdm_asyncio.gather(*tasks, desc=f"Translating batch {i//batch_size + 1}")

            batch_translated_text = '\n'.join(batch_translated_chunks)
            translated_text += batch_translated_text + '\n\n'

            # Periodic progress update
            if (i//batch_size + 1) % 5 == 0 or i + batch_size >= total_pages:
                logging.info(f"Progress: {min(i + batch_size, total_pages)}/{total_pages} pages translated")

            # Add a delay between batches to limit the rate
            await asyncio.sleep(5)

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