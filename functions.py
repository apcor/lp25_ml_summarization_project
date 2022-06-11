import re

import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


def parse_wiki(url) -> str:
    wiki_parsed = BeautifulSoup(requests.get(url).text, 'html.parser')
    paragraphs = wiki_parsed.find_all('p')
    article_content = ''
    for p in paragraphs:  
        article_content += re.sub(r"\[[\S]*\]", "", p.text)
    return article_content


def create_dictionary_table(text_string) -> dict:
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    stem = PorterStemmer()
    
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table


def calculate_sentence_scores(sentences, frequency_table) -> dict:
    sentence_weight = dict()
    for sentence in sentences:
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence[:15] in sentence_weight:
                    sentence_weight[sentence[:15]] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence[:15]] = frequency_table[word_weight]

    sentence_weight[sentence[:15]] = sentence_weight[sentence[:15]] / sentence_wordcount_without_stop_words
    return sentence_weight

def calculate_average_score(sentence_weight) -> int:
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]
    average_score = (sum_values / len(sentence_weight))

    return average_score

def get_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''

    for sentence in sentences:
        if sentence[:15] in sentence_weight and sentence_weight[sentence[:15]] >= (threshold):
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary
