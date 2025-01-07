from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from mecab import MeCab

from embedding.word2vec import get_example_doc

raw_text_list = get_example_doc()
mecab = MeCab()

test_text = []
for text_list in raw_text_list:
    test_text.append(mecab.nouns("".join(text_list)))


dictionary = Dictionary(test_text)
id2token = {v: k for k, v in dictionary.token2id.items()}
corpus = [dictionary.doc2bow(text) for text in test_text]


coh_list = []
for num_topics in range(2, 11):
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2token, passes=10)
    coh_model = CoherenceModel(model=lda_model, texts=test_text, corpus=corpus, dictionary=dictionary, coherence="c_v")
    coh = coh_model.get_coherence()
    print(num_topics, coh)
    coh_list.append(coh)

best_topics = coh_list.index(max(coh_list)) + 2

lda_model = LdaModel(corpus=corpus, num_topics=best_topics, id2word=id2token, passes=10)
coh_model = CoherenceModel(model=lda_model, texts=test_text, corpus=corpus, dictionary=dictionary, coherence="c_v")
lda_model.print_topics(-1)
