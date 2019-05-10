import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
CUR_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

sys.path.append(CUR_DIR + 'genetic_algorithm/')
from learn_population import  learn_population
from create_population import create_population

def loadRanks(trecks):
    filename = CUR_DIR + 'Data/ranks' + str(trecks) + '.txt'
    return pd.read_csv(filename, delimiter=' ', names=['QueryId', 'smth', 'document', 'tag'])

def loadMetadata(filename):
    lines = map(lambda x: x.strip().split(' '), open(filename, 'r').readlines())
    return dict(map(lambda x: (x[0], float(x[1])), lines))

def isMember(arr, to_remain):
    result, indexes = [], []
    for i, elem in enumerate(arr):
        result.append(elem in to_remain)
        if result[-1]:
            indexes.append(np.where(to_remain == elem)[0][0])
            
    return np.array(result), np.array(indexes)

def loadCode(ranks, qId, qIndex, trecks):
    metadata = loadMetadata(CUR_DIR + 'Data/data/metadata' + str(trecks) + '.txt')
    
    doc_id_string = pd.read_csv(CUR_DIR + 'Data/Data{0}/doc_id_strings{1}.txt'.format(trecks, qIndex),
                               delimiter=' ',
                               names=['Name', 'Id'])
    
    term_doc_vars = pd.read_csv(CUR_DIR + 'Data/Data{0}/termvars{1}.txt'.format(trecks, qIndex),
                               delimiter=' ',
                               names=['Q1', 'Q2', 'Q3'])
    
    lengthForTerms = np.array(list(map(lambda x: int(x.strip()),
                         open(CUR_DIR + 'Data/Data{0}/lengthForTerms{1}.txt'.format(trecks, qIndex), 'r').readlines())))
    
    q_ranks = ranks[ranks['QueryId'] == qId]
    docs_to_retrive = q_ranks['document'].values 
    
    # begin of my debug code
    #print("doc_values")
    #print(doc_id_string['Name'].values)
    #print("docs_to_retrive")
    #print(docs_to_retrive)
    # end of my debug code
    
    inds_to_retrive, _ = isMember(doc_id_string['Name'].values, docs_to_retrive)
    
    doc_id_retrive = doc_id_string[inds_to_retrive]
    ids_to_retrive = doc_id_retrive['Id'].values
    
    # begin of my debug code
    #print("doc_values")
    #print(doc_id_retrive)
    #print("docs_to_retrive")
    #print(ids_to_retrive)
    # end of my debug code
    
    start = 0
    vecDelimCopy = np.ones(lengthForTerms.shape)
    matTermDocVarsCopy = None

    for i in np.arange(lengthForTerms.shape[0]):
        
        slice_to_work_with = term_doc_vars[start:start+lengthForTerms[i]]
        vec_from_slice = slice_to_work_with[slice_to_work_with.columns[0]].values
        
        inds_to_remain, _ = isMember(vec_from_slice, ids_to_retrive)
        slice_to_work_with = slice_to_work_with.iloc[inds_to_remain]
        
        if matTermDocVarsCopy is None:
            matTermDocVarsCopy = slice_to_work_with
        else:
            matTermDocVarsCopy = pd.concat([matTermDocVarsCopy, slice_to_work_with])
        vecDelimCopy[i] = len(inds_to_remain)        
        start += lengthForTerms[i]
    
    matTermDocVars = matTermDocVarsCopy
    vecDelim = vecDelimCopy.T

    return (metadata, doc_id_retrive, q_ranks, matTermDocVars, vecDelim)
    
def rewriteRanks(vecNamesDocRanks, matDocIdStr):
    vecNamesIdStr = matDocIdStr.iloc[:, 0].values
    vecIdsIdStr = matDocIdStr.iloc[:, 1].values
    
    _, indInNamesIdStr = isMember(vecNamesDocRanks, vecNamesIdStr)
    
    if len(indInNamesIdStr):
        idsOfDocRanks = vecIdsIdStr[indInNamesIdStr]
    else:
        idsOfDocRanks = []
        
    return np.sort(idsOfDocRanks)

def loadData(trecks, doc_cluster_names = None):
    ranks = loadRanks(trecks)
    #print("ranks\n\n", ranks)
    
    in_cluster_indices = np.arange(ranks.shape[0])
    if not (doc_cluster_names is None):
        is_doc_from_cluster = lambda doc_name: doc_name in doc_cluster_names
        in_cluster_indices = np.vectorize(is_doc_from_cluster)(ranks['document'])
        ranks = ranks[in_cluster_indices]
        
    #print("relevant ranks\n\n", ranks)

    
    queries = pd.read_csv(CUR_DIR + 'Data/data/queries' + str(trecks) + '.txt', 
                          delimiter='#',
                          names=['Id', 'Query'])

    modelCharacteristics = [None for i in range(queries.values.shape[0])]
    mat_doc_ranks = []
    is_query_nonempty = np.zeros(len(queries))
    
    for i, query_id in enumerate(queries['Id'].values):

        print("Query : ", query_id)

        matMetaData, matDocIdStr, matDocRanks, matTermDocVars, vecDelim = loadCode(ranks, query_id, i + 1, trecks)
        
        
        AvDocLen = matMetaData['AverageDocumentLength']
        NumbDocs = matMetaData['NumberOfDocuments']

        ulabel, uindex = np.unique(matTermDocVars[matTermDocVars.columns[0]], return_inverse=True)
        modelCharacteristics[i] = [ulabel, uindex]        
        
        xvars = matTermDocVars.iloc[:,1].values * np.log10((AvDocLen + matTermDocVars.iloc[:, 2].values) / matTermDocVars.iloc[:, 2].values)
        
  
        yvars = vecDelim / NumbDocs
        
        yvarsExt_ed = np.repeat(yvars, vecDelim.astype(np.int32))
        modelCharacteristics[i].append([xvars, yvarsExt_ed])
        
        cur_doc_ranks = [[], [], [], [], []]
        
        for j in range(4):
            cur_doc_ranks[j] = matDocRanks.iloc[:, j]
        
        vecDocNamesEv  = matDocRanks.iloc[:, 2].values
        vecDocRanksEv  = matDocRanks.iloc[:, 3].values
        cur_doc_ranks[2]  = vecDocNamesEv[vecDocRanksEv == 1]
        
        if len(cur_doc_ranks[2]):
            cur_doc_ranks[4] = rewriteRanks(cur_doc_ranks[2], matDocIdStr)
            mat_doc_ranks.append(cur_doc_ranks)
            is_query_nonempty[i] = 1
            
    stop = True
    is_query_nonempty = is_query_nonempty.astype(bool)
    return (mat_doc_ranks, 
            queries.loc[is_query_nonempty,], 
            np.array(modelCharacteristics)[is_query_nonempty], )

#doc_ranks, queries, query_characteristics = loadData(6)



# learn_population.test(doc_ranks, queries, query_characteristics)


#population = create_population.create_population(10, 10)
#learn_population.learn_population(population, doc_ranks, queries, query_characteristics)

