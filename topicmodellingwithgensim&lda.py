from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import spacy
import pyLDAvis.gensim

from google.colab import files
uploaded = files.upload()

data = pd.read_csv("Bishop.csv")

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
  doc = nlp(text)
  tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
  return tokens

data.head()

processed_docs = [preprocess(text) for text in data["topic"]]
dictionary = corpora.Dictionary()
dictionary.add_documents(processed_docs)

corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
lda_model = LdaModel(corpus= corpus, id2word=dictionary, num_topics=5, passes=10)
for topic_id, topic in lda_model.print_topics():
  print(f"Topic {topic_id + 1}: {topic}")
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
