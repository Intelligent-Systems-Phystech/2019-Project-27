import numpy as np

def isMember(arr, to_remain):
    result, indexes = [], []
    for i, elem in enumerate(arr):
        result.append(elem in to_remain)
        if result[-1]:
            indexes.append(np.where(to_remain == elem)[0][0])
            
    return np.array(result, dtype=np.int), np.array(indexes, dtype=np.int)


def evaluate_model(model, query_characteristics):
    values = model.calc(*query_characteristics[2])
    result = np.zeros(query_characteristics[0].shape[0])
    for i, pos in enumerate(query_characteristics[1]):
        result[pos] += values[i]

    return np.array([query_characteristics[0], result])

def evaluate_quality(query_id, doc_related, doc_ranks):
    #print(query_id, doc_related, doc_ranks)
    vecDocIdEv = doc_ranks[4]
    
    sort_indexes = np.argsort(doc_related[0])
    doc_related = doc_related[:, sort_indexes]
    
    indsWhichAppear, _ = isMember(doc_related[0], vecDocIdEv)
    doc_related = np.vstack((doc_related, indsWhichAppear))

    sort_indexes = np.argsort(-doc_related[1], kind='mergesort')
    doc_related = doc_related[:, sort_indexes]

    ranksForRetrievedDocs = doc_related[2]
    cumRanksForRetrievedDocs = np.cumsum(ranksForRetrievedDocs)
    cutOffPrecision = cumRanksForRetrievedDocs / np.arange(1, ranksForRetrievedDocs.shape[0] + 1)
    qualValue = np.sum(cutOffPrecision * ranksForRetrievedDocs) / doc_ranks[2].shape[0]
    
    return qualValue

def get_quality(model, doc_ranks, queries, query_characteristics):

    vec_quality = []
    for i, query_id in enumerate(queries['Id'].values):
        doc_related = evaluate_model(model, query_characteristics[i])
        quality = evaluate_quality(query_id, doc_related, doc_ranks[i])
        vec_quality.append(quality)
    
    #print(np.sum(np.round(vec_quality, 4)))

    quality = np.sum(vec_quality)

    # TODO add regularization 
    return quality