from nepalikit.tokenization import Tokenizer
from nepalikit.preprocessing import TextProcessor
from nepalikit.manage_stopwords import load_stopwords, remove_stopwords_from_text
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "stopwords")

processor = TextProcessor()
tokenizer = Tokenizer()
stopwords = load_stopwords(file_path)

def preprocess_text(text:str):
    clean_text = processor.remove_html_tags(text)
    clean_text = processor.remove_special_characters(clean_text)
    clean_text = remove_stopwords_from_text(clean_text,stopwords)
    tokens = tokenizer.tokenize(clean_text, level='word')
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return tokenizer.detokenize(tokens)