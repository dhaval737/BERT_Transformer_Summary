import re

from cleantext import clean
import ftfy
from nltk import tokenize

from collections import Counter

def sentence_check(sentence):
    n_upper = sum(map(str.isupper, sentence))
    n_lower = sum(map(str.islower, sentence))
    perc_up = n_upper / (n_upper + n_lower + 1)
    alphanum = len("".join(filter(str.isalnum, sentence)))
    prc_alphanum = alphanum / ((len(sentence)) + 1)
    return perc_up < 0.5 and prc_alphanum > 0.5
def clean_file(text):
    '''clean file with ad-hoc rules'''
    # clean page numbers
    text = " ".join([line for line in text.splitlines() if not line.strip().isdigit()])
    # clean new lines
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    # clean sequence of 3+ symbols, substitute with one only
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # text = text.encode("ascii", "ignore").encode("ascii", "ignore")
    text = ftfy.fix_encoding(text)

    # clean pattern - number followed by letter, letter followed by number
    text = re.sub(r'([0-9$%€£)]{1})([A-Za-z]{1})', r'\1 \2', text)
    text = re.sub(r'([A-Za-z]{1})([(0-9$%€£]{1})', r'\1 \2', text)
    # clean from consecutive 4+ all upper cases words and numbers,
    text = re.sub(r'((\b[A-Z0-9]{2,}\b ){4,})', r'', text)
    # clean from consecutive 5 numbers. no!!!
    # text = re.sub(r'((\b[0-9]{1,}\b ){5,})', r'', text)
    # consecutive not alpha numeric symbols
    text = re.sub('([^0-9a-zA-Z]+ ){3,}', '', text)
    # split in sentences
    sentences = tokenize.sent_tokenize(text)

    sentences = [s for s in sentences if sentence_check(s)]

    s_count = Counter(sentences)
    sentences = [s for s in sentences if s_count[s] == 1]
    text = " ".join(sentences)
    text = re.sub(' +', ' ', text)

    # clean different things
    text = clean(text#,
                 #fix_unicode=True,  # fix various unicode errors
                 #to_ascii=True,  # transliterate to closest ASCII representation
                 #lower=True,  # lowercase text
                 #no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                 #no_urls=False,  # replace all URLs with a special token
                 #no_emails=True,  # replace all email addresses with a special token
                 #no_phone_numbers=True,  # replace all phone numbers with a special token
                 #no_numbers=False,  # replace all numbers with a special token
                 #no_digits=False,  # replace all digits with a special token
                 #no_currency_symbols=False,  # replace all currency symbols with a special token
                 #no_punct=False,  # remove punctuations
                 #replace_with_punct="",  # instead of removing punctuations you may replace them
                 #replace_with_url="",
                 #replace_with_email="",
                 #replace_with_phone_number="",
                 #replace_with_number="<NUMBER>",
                 #replace_with_digit="0",
                 #replace_with_currency_symbol="<CUR>",
                 #lang="en"  # set to 'de' for German special handling
                 )

    return text