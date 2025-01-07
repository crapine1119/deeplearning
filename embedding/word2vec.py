import gensim
import numpy as np
from gensim.corpora import Dictionary
from konlpy.corpus import kobill, kolaw  # Docs from pokr.kr/bill
from mecab import MeCab

# model = embedding.models.KeyedVectors.load_word2vec_format(
#     "/Users/songhak/model/GoogleNews-vectors-negative300.bin.gz", binary=True
# )

model = gensim.models.Word2Vec.load("/Users/songhak/model/embedding/word2vec/ko/ko.bin")
mecab = MeCab()

sentences = [
    mecab.morphs("나는 밥을 먹으러 간다"),
    mecab.morphs("나는 간다 밥을 먹으러"),
    mecab.morphs("나는 밥을 먹으러 가지 않는다"),
]
vector_size = model.vector_size
sentence_vectors = []
for sentence in sentences:
    sentence_vector = []
    for token in sentence:
        vector_info = model.wv.vocab.get(token, None)
        if vector_info is None:
            vector = np.zeros(vector_size)
        else:
            vector = model[token]
        sentence_vector.append(vector)
    sentence_vectors.append(np.stack(sentence_vector))
##


def get_example_doc():
    files_bill = kobill.fileids()  # Get file ids
    files_law = kolaw.fileids()  # Get file ids

    text_bill = [mecab.morphs(kobill.open(f).read()) for f in files_bill]
    text_law = [mecab.morphs(kolaw.open(f).read()) for f in files_law]
    total_text = text_bill + text_law + [["나", "는", "밥", "을", "먹", "으러", "간다"]]
    return total_text


total_text = get_example_doc()
dictionary = Dictionary(total_text)
# 메모리때문에 자동으로 생성을 안해줌
id2token = {i: token for token, i in dictionary.token2id.items()}
# 문서-단어 행렬(document-term matrix) 생성
corpus = [dictionary.doc2bow(text) for text in total_text]
tfidf = gensim.models.TfidfModel(corpus)

id2w = {}
for c in tfidf[corpus]:
    id2w.update(dict(c))

##
sentence_tfidf_weights = []
for sentence, sentence_v in zip(sentences, sentence_vectors):
    ids = [dictionary.token2id.get(c, None) for c in sentence]
    w = np.array([id2w.get(i, 0) for i in ids])
    weighted_avg = np.einsum("ab,a->ab", sentence_v, w)
    weighted_avg.mean(0).shape
    sentence_tfidf_weights.append(weighted_avg.mean(0))


def cos_sim(a, b):
    return (a @ b) / np.linalg.norm(a) / np.linalg.norm(b)


cos_sim(sentence_tfidf_weights[0], sentence_tfidf_weights[1])
cos_sim(sentence_tfidf_weights[0], sentence_tfidf_weights[2])

cos_sim(sentence_vectors[0].mean(0), sentence_vectors[1].mean(0))
cos_sim(sentence_vectors[0].mean(0), sentence_vectors[2].mean(0))
