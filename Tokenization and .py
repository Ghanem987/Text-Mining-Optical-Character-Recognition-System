import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt_tab")
nltk.download("stopwords")
text = "The quick brown fox jumps over the lazy dog while the sun is shining and the birds are singing ."
tokens = word_tokenize(text)
print("tokens: ",tokens)
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.upper() not in stop_words]
print("filtered tokens : ",filtered_tokens)