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
    
    inds_to_retrive, _ = isMember(doc_id_string['Name'].values, docs_to_retrive)
    
    doc_id_retrive = doc_id_string[inds_to_retrive]
    ids_to_retrive = doc_id_retrive['Id'].values
    
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

    idsOfDocRanks = vecIdsIdStr[indInNamesIdStr]

    return np.sort(idsOfDocRanks)

def loadData(trecks):
    ranks = loadRanks(trecks)
    
    queries = pd.read_csv(CUR_DIR + 'Data/data/queries' + str(trecks) + '.txt', 
                          delimiter='#',
                          names=['Id', 'Query'])

    modelCharacteristics = [None for i in range(queries.values.shape[0])]
    mat_doc_ranks = [[None, None, None, None, None] for i in range(queries.values.shape[0])]

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
        
        for j in range(4):
            mat_doc_ranks[i][j] = matDocRanks.iloc[:, j]
        
        vecDocNamesEv  = matDocRanks.iloc[:, 2].values
        vecDocRanksEv  = matDocRanks.iloc[:, 3].values
        mat_doc_ranks[i][2]  = vecDocNamesEv[vecDocRanksEv == 1]
        mat_doc_ranks[i][4] = rewriteRanks(mat_doc_ranks[i][2], matDocIdStr)
    stop = True
    return (mat_doc_ranks, queries, modelCharacteristics, )

#doc_ranks, queries, query_characteristics = loadData(6)



# learn_population.test(doc_ranks, queries, query_characteristics)


#population = create_population.create_population(10, 10)
#learn_population.learn_population(population, doc_ranks, queries, query_characteristics)

