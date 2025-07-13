import nltk
from nltk.corpus import wordnet as wn
import scipy.cluster.hierarchy as sch
import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt

# download WordNet
nltk.download('wordnet')

# wup_similarity
def wordnet_similarity(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 or not synsets2:
        return 0  
    
    max_sim = 0
    for syn1 in synsets1:
        for syn2 in synsets2:
            sim = syn1.wup_similarity(syn2) 
            if sim is not None and sim > max_sim:
                max_sim = sim
    
    return max_sim

# vg150 predicate
categories = ["above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]


num_categories = len(categories)
similarity_matrix = np.zeros((num_categories, num_categories))

for i in range(num_categories):
    for j in range(i, num_categories):
        sim = wordnet_similarity(categories[i], categories[j])
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim 


print("Similarity Matrix:")
print(similarity_matrix)

# Hierarchical clustering
distance_matrix = 1 - similarity_matrix

np.fill_diagonal(distance_matrix, 0)
linkage_matrix = sch.linkage(dist.squareform(distance_matrix), method='ward')


# set threshold
max_d = 0.3  
clusters = sch.fcluster(linkage_matrix, max_d, criterion='distance')

print("\nCluster Assignment:")
for i, category in enumerate(categories):
    print(f"Category: {category}, Cluster: {clusters[i]}")

unique_clusters = np.unique(clusters)
for cluster in unique_clusters:
    print(f"\nCluster {cluster}:")
    for i, category in enumerate(categories):
        if clusters[i] == cluster:
            print(f"  - {category}")
