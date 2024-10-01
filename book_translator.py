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
openai.api_key = OPENAI_API_KEY

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
        
        logging.info(f"Extracted {len(full_text.split('\f'))} pages from PDF")
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

async def translate_chunk(session, chunk: str, source_lang: str, target_lang: str, model: str, retries: int = 3) -> str:
    logging.info(f"Translating chunk: {chunk[:100]}...")  # Log first 100 characters of the chunk

    for attempt in range(retries):
        try:
            if model == 'google_gemini':
                print(f"Translating chunk with Google Gemini: {chunk[:100]}...")
                translated_chunk = await google_gemini_translate(session, chunk, source_lang, target_lang)
            elif model == 'openai':
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

async def openai_translate(text: str, source_lang: str, target_lang: str) -> str:
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, openai.ChatCompletion.create,
        {
            'model': "gpt-4",
            'messages': [
                {"role": "system", "content": f"You are a translator proficient in {source_lang} and {target_lang}."},
                {"role": "user", "content": f"Please translate the following text from {source_lang} to {target_lang}:\n\n{text}"}
            ]
        }
    )
    return response['choices'][0]['message']['content'].strip()

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

    if not os.path.exists(full_path):
        logging.error(f"os.path.exists() says the file {full_path} does not exist.")
    else:
        logging.info(f"os.path.exists() says the file {full_path} exists.")

    if not os.access(full_path, os.R_OK):
        logging.error(f"os.access() says the file {full_path} is not readable.")
    else:
        logging.info(f"os.access() says the file {full_path} is readable.")

    if not os.path.exists(full_path) or not os.access(full_path, os.R_OK):
        logging.error(f"Cannot access the file {full_path}.")
        return

    logging.info("Extracting text from PDF...")
    try:
        full_text = extract_text_from_pdf(args.pdf_path)
        
        # Split the full text into pages (assuming each page ends with a form feed character '\f')
        pages = full_text.split('\f')
        
        # Process the book in batches of 5 pages
        batch_size = 5
        translated_text = ""
        
        for i in range(0, len(pages), batch_size):
            batch_pages = pages[i:i+batch_size]
            batch_text = '\f'.join(batch_pages)
            
            print(f"\nProcessing batch {i//batch_size + 1} (pages {i+1}-{min(i+batch_size, len(pages))}):")
            print("------------------------")
            print(batch_text[:1000] + "..." if len(batch_text) > 1000 else batch_text)
            print("------------------------")
            
            user_input = input("Does this batch of text look correct? (y/n): ")
            if user_input.lower() != 'y':
                logging.error(f"Text extraction quality check failed for batch {i//batch_size + 1}. Skipping this batch.")
                continue

            logging.info(f"Translating batch {i//batch_size + 1}...")
            sentences = split_text_into_sentences(batch_text)
            chunks = chunk_text(sentences, args.max_tokens - 500 if args.max_tokens else 1548, args.overlap)

            async with aiohttp.ClientSession() as session:
                tasks = [translate_chunk(session, chunk, args.source_lang, args.target_lang, args.model, args.retries) for chunk in chunks]
                batch_translated_chunks = await tqdm_asyncio.gather(*tasks, desc=f"Translating batch {i//batch_size + 1}")

            batch_translated_text = '\n'.join(batch_translated_chunks)
            translated_text += batch_translated_text + '\n\n'  # Add extra newlines between batches

            print(f"\nSample translation for batch {i//batch_size + 1}:")
            print(batch_translated_text[:500] + "..." if len(batch_translated_text) > 500 else batch_translated_text)
            user_input = input("Does this batch translation look correct? (y/n): ")
            if user_input.lower() != 'y':
                logging.warning(f"Translation quality check failed for batch {i//batch_size + 1}. The translation may need review.")

        # Save the complete translated text
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