from nltk.corpus import stopwords
import re

def stopwordsfilter(text, **argv):
    lang = argv.get('lang', 'en')
    if lang == 'en': lang = 'english' 

    words = []
    stops = set(stopwords.words(lang))
    for word in re.split('\s+', text):
        if word.lower() not in stops:
            words.append(word)

    return ' '.join(words)

def lowercase(text, **argv):
    return text.lower()