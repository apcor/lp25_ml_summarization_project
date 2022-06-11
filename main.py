"""
Source and inspiration:
https://blog.floydhub.com/gentle-introduction-to-text-summarization-in-machine-learning/
"""

from functions import (create_dictionary_table, calculate_average_score, 
calculate_sentence_scores, get_article_summary, parse_wiki)
from nltk.tokenize import sent_tokenize


def run_article_summary(article_content):
    
    #creating a dictionary for the word frequency table
    frequency_table = create_dictionary_table(article_content)

    #tokenizing the sentences
    sentences = sent_tokenize(article_content)

    #algorithm for scoring a sentence by its words
    sentence_scores = calculate_sentence_scores(sentences, frequency_table)

    #getting the threshold
    threshold = calculate_average_score(sentence_scores)

    #producing the summary
    article_summary = get_article_summary(sentences, sentence_scores, 1.5 * threshold)

    return article_summary


if __name__ == '__main__':
    wiki_url = input(f"Please enter URL of an English Wikipedia article to summarize. Type 'No' for demonstration.\n")
    if wiki_url.lower() == 'no':
        with open('NLP_article_test.txt', 'r', encoding='utf-8') as f:
            article_content = f.read()
        summary_results = run_article_summary(article_content)
    else:
        summary_results = run_article_summary(parse_wiki(wiki_url))
    print(f"Have a look at this article's summary below (extraction-based summarization method):\n\n{summary_results}")