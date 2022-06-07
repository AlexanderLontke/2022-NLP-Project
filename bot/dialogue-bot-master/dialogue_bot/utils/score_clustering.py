import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, SpectralClustering


def top_similar_scores(lst, min_score=None, key=lambda l: l, alg=AgglomerativeClustering(n_clusters=2)):
    if len(lst) <= 1:
        return lst

    scores = [{'score': key(x), 'obj': x, 'orig': True} for x in lst]
    if min_score is not None:
        scores.append({'score': min_score, 'obj': None, 'orig': False})
    scores = sorted(scores, key=lambda x: x['score'], reverse=True)

    X = np.array([x['score'] for x in scores]).reshape(-1, 1)
    alg.fit(X)
    labels = alg.labels_
    return [x['obj'] for x, l in zip(scores, labels) if x['orig'] and l == labels[0]]


def plot_experiments(n, min_score, max_score, insert_scores=None, nr_experiments=10):
    if insert_scores is None:
        insert_scores = []

    color_range = ['blue', 'red', 'green', 'brown', 'black', 'yellow']
    clustering_algs = [
        ('KMeans', KMeans(n_clusters=2, random_state=0)),
        ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=2)),
        ('SpectralClustering', SpectralClustering(n_clusters=2)),
        ('MeanShift', MeanShift()),
    ]

    fig, axs = plt.subplots(nr_experiments, len(clustering_algs), figsize=(len(clustering_algs)*2, nr_experiments*2))
    for i in range(nr_experiments):

        # create random data
        scores = [random.uniform(min_score, max_score) for _ in range(n)]
        for score in insert_scores:
            scores.append(score)
        scores = sorted(scores, reverse=True)
        n_scores = len(scores)

        X = np.array(scores).reshape(-1, 1)

        # apply clustering algorithms
        for j in range(len(clustering_algs)):
            alg_name, alg = clustering_algs[j]
            alg.fit(X)
            labels, clusters = alg.labels_, set(alg.labels_)

            axs[i, j].set_title(alg_name)
            for cluster in clusters:
                color = color_range[cluster % len(color_range)]
                axs[i, j].scatter([i for i in range(n_scores) if labels[i] == cluster], [scores[i] for i in range(n_scores) if labels[i] == cluster], c=color)

    plt.show()


if __name__ == '__main__':
    # plot_experiments(2, 0.75, 1, insert_scores=[0.75])
    plot_experiments(10, 0.75, 1)
    # print(top_similar_scores([
    #     0.9704794937460413,
    #     0.9594906136108781,
    #     0.9565593760816187,
    #     0.9481323498893024,
    #     0.9406653510855512,
    #     0.8635669552981207,
    #     0.8214570780219523,
    #     0.787414568703562,
    #     0.7538717600527693,
    #     0.7500376004936383])
    # )
    print(top_similar_scores([0.97, 0.90], min_score=0.75))
