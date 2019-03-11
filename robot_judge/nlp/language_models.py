import spacy
import nltk
from nltk.corpus import stopwords

# Load spacy language model and nltk english stopwords
spacy_nlp = spacy.load('en')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
