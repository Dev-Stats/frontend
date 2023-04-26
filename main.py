from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx


def read_article(filename: str):
    file = open(filename, "r")
    filedata = file.read()
    article = filedata.split(".")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences


def sentence_similarity(sentence1, sentence2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sentence1 = [w.lower() for w in sentence1]
    sentence2 = [w.lower() for w in sentence2]

    all_words = list(set(sentence1+sentence2))
    vector1 = [0]*len(all_words)
    vector2 = [0]*len(all_words)
    for w in sentence1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sentence2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return cosine_similarity([vector1], [vector2])[0][0]


def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = sentence_similarity(
                sentences[i], sentences[j], stop_words)
    return similarity_matrix


def generate_summary(filename, top_n=10):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences = read_article(filename)

    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    print("Summarize Text: \n", ". ".join(summarize_text))


generate_summary("tigercat.txt", 10)
