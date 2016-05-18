import nltk
from nltk.tokenize import WhitespaceTokenizer


tokenizer = WhitespaceTokenizer()

nltk.download()

from nltk.corpus import stopwords
stopword_list = stopwords.words('english')
print stopword_list 

import pandas as pd
reviews_df = pd.read_csv(r"C:\Users\Deon\test_chunk1.csv", encoding="utf-8") #replace with smaller chunks
#reviews_df 

positive_terms = []
f = open('C:\Users\Deon\Downloads\positive_terms.txt', "r")
positive_terms = f.read().splitlines()
f.close()

negative_terms = []
f = open(r'C:\Users\Deon\Downloads\negtive_terms.txt', "r")
negative_terms = f.read().splitlines()
f.close()

print positive_terms
print ""
print negative_terms

porter = nltk.PorterStemmer()

import string

def remove_punctuation(text):
    #this removes punctuations that are defined in the string library
    punctuations = string.punctuation 

    # But don't strip out apostrophes
    excluded_punctuations = ["'"]
    for p in punctuations:
        if p not in excluded_punctuations:
            # replace each punctuation symbol by a space
            text = text.replace(p, " ") 

    return text

porter = nltk.PorterStemmer()


def normalize_review_text(text):
    #text = text.lower()
    text = remove_punctuation(text)
    text = " ".join(text.split())
    text_tokens = tokenizer.tokenize(text)
    text_tokens = [porter.stem(w) for w in text_tokens if w not in stopword_list]
    return text_tokens

# Apply the function above to the text column
reviews_df["text"] = reviews_df["text"].apply(normalize_review_text)
reviews_df

def calculate_positivity(text):
    num_tokens = len(text)
    num_positive_tokens = 0
    for t in text:
        if t in positive_terms:
            num_positive_tokens = num_positive_tokens + 1
    # The positivity score is the fraction of tokens that were found in the positive dictionary
    return float(num_positive_tokens) / float(num_tokens)

reviews_df["positivity_score"] = reviews_df["text"].apply(calculate_positivity)

def calculate_negativity(text):
    num_tokens = len(text)
    num_negative_tokens = 0
    for t in text:
        if t in negative_terms:
            num_negative_tokens = num_negative_tokens + 1
    # The positivity score is the fraction of tokens that were found in the positive dictionary
    return float(num_negative_tokens) / float(num_tokens)

reviews_df["negativity_score"] = reviews_df["text"].apply(calculate_negativity)
reviews_df

import scipy.stats as sp

pos_score_stars_corr = sp.pearsonr(reviews_df["stars"].values, reviews_df["positivity_score"].values)
pos_score_stars_corr

print ""

neg_score_stars_corr = sp.pearsonr(reviews_df["stars"].values, reviews_df["negativity_score"].values)
neg_score_stars_corr
