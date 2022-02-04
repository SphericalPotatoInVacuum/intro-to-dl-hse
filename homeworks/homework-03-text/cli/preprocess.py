from loguru import logger
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import string

import nltk


def process_text(text: str) -> list[str]:
    # remove empty parts
    text = text.replace('No Negative', '').replace('No Positive', '')
    # tokenize
    words = word_tokenize(text)
    # remove all punctuation
    words = [word.strip(string.punctuation) for word in words]
    # remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    # stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # remove small words
    words = [word for word in words if len(word) > 1]
    return words


def preprocess(input_path: str, output_path: str):
    from pandarallel import pandarallel
    pandarallel.initialize()

    nltk.download('punkt')
    nltk.download('stopwords')

    logger.info(f'Processing file {input_path}')

    df = pd.read_csv(input_path)

    df['negative'] = df['negative'].parallel_apply(process_text)
    df['positive'] = df['positive'].parallel_apply(process_text)
    df['review'] = df['positive'] + df['negative']

    # to_pickle to save numpy arrays, otherwise everything will turn to strings
    df.to_pickle(output_path)

    logger.success(f'Saved processed file at {output_path}')
