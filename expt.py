import networkx as nx
from scipy.spatial import distance
import lutnet
import logging
import pickle
import itertools
import librosa
from datetime import datetime
import os
import sys
import copy
import ray
ray.init()
from scipy.stats import rv_discrete
import math
import numpy 

#%pylab equiv
import numpy
np = numpy
from pylab import *
from numpy import *

from sklearn.model_selection import train_test_split



def pdm(x, oversample=1):
    n = len(x)
    y = [0 for x in range(n * oversample)]
    idx=0
    error = [0 for x in range(len(y) + 1)]
    for i in range(n):
        for j in range(oversample):
            y[idx] = 1 if x[i] >= error[idx] else 0
            error[idx+1] = y[idx] - x[i] + error[idx]
            idx += 1
    return y, error[0:n]




def ttable_gen_linrand():
    return random.randint(2, size=16)

def ttable_gen_sparse1():
    x = zeros(16, dtype=int)
    x[random.randint(16)] = 1
    return x

    
def makeLUT4(inputs, layer, idx, ttable_gen=ttable_gen_linrand):
    lut4 = {'layer':layer, 'idx':idx, 'out':0, 'ttable':ttable_gen(), 'netIn':inputs}
    return lut4


def byteToBits(val, bitCount):
    return [int(val & (1 << (bitCount-x-1)) > 0) for x in range(bitCount)]

def bitsToInt(bits):
    return sum([x * pow(2,len(bits)-i-1) for i,x in enumerate(bits)])
 
def calcNet(lutnet):
    for layer in lutnet[1:]:
        calcLayer(layer)
        
hamming4bit = []
for i in range(16):
    dists = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for j in range(16):
        dists[j] = distance.hamming(byteToBits(i, 4), byteToBits(j,4))
    hamming4bit.append(dists)
        


def byteToBits4(x):
    return byteToBits4Table[x]

def dataToBinary(data, base=8):
    xmin = np.min(data[:,0])
    xmax = np.max(data[:,0])    
    quantData = [[np.round(np.interp(x[0],[xmin, xmax],[0,pow(2,base)-1]))] for x in data]
    return np.array([byteToBits(int(x[0]), base) for x in quantData])

def dataToBCD(data, decimalPlaces=2):
    xmin = np.min(data[:,0])
    xmax = np.max(data[:,0])    
    ymin = np.min(data[:,1])
    ymax = np.max(data[:,1])    
    print(xmin,xmax, ymin, ymax)
    normMax = 1.0 - 1e-16
    normData = [[np.interp(x[0],[xmin, xmax],[0,normMax]),
             np.interp(x[1],[ymin, ymax],[0,normMax])] for x in data]
    print(array(normData))
    def bcdize(d):
        binary = zeros(0, dtype=int)
        for p in range(decimalPlaces):
            d = d * 10
            digit = int(d)
            binary = np.concatenate((binary, byteToBits(digit,4)))
            d = d - digit
        return binary
        
    quantData = np.array([np.concatenate((bcdize(x[0]),bcdize(x[1]))).tolist() for x in normData])
    return quantData

def runRC(rc, inputs):    
    nodeMap = rc['nodes'].copy()
    nodeMap.extend(inputs)
    for node in range(rc['N']):
        lookupAddress=0;
        for inputIdx in range(len(rc['net'][node])):
            lookupAddress += nodeMap[rc['net'][node][inputIdx]] * rc['binMult'][inputIdx]
        rc['nodes'][node] = rc['lookups'][node][lookupAddress]
    return rc

def rbnToNetX(rbn):
    G = nx.DiGraph()
    nodesMap = []
    for i_node in range(rbn['N']):        
        for i_input in rbn['net'][i_node]:
            if (i_input < rbn['N']):
                edge = (i_input,i_node)
                G.add_edge(*edge)
    return G

from enum import Enum
class monolithicRBNInputSchemes(Enum):
    LINRAND = 0
    BTWCENTRALITYHIGH = 1
    BTWCENTRALITYLOW = 2

class monolithicRBNOutputSchemes(Enum):
    LINRAND = 0
    BTWCENTRALITYHIGH = 1
    BTWCENTRALITYLOW = 2

    
def makeRBN(N=16, pk=[0.1,0.5,0.3,0.1], externalInputConnectivity = 0.3, 
            externalInputAllocScheme = monolithicRBNInputSchemes.LINRAND,
            nIns=1, 
            externalOutputAllocScheme = monolithicRBNOutputSchemes.LINRAND,
            outputSize=0.1):
    custm = rv_discrete(name='custm', values=([1,2,3,4], pk))
    RBN = {'N':N, 'pk':pk, 'p':0.5, 'nIns': nIns, 'extIns':externalInputConnectivity}
    RBN['nodes'] = [random.random() > RBN['p'] for x in range(N)] 
    RBN['lookups'] = [[random.random() > RBN['p'] for ltentry in range(16)] for node in range(N)]  
    RBN['binMult'] = [1,2,4,8]
    RBN['net'] = [[]] * N  

    inTotal=0
    for node in range(N):
        inCount = custm.rvs()
        inTotal = inTotal + inCount
        inputs = [np.random.randint(0, RBN['N']) for x in range(inCount)]
        RBN['net'][node] = inputs

    G = btw = ps = None
    if externalInputAllocScheme != monolithicRBNInputSchemes.LINRAND or externalOutputAllocScheme != monolithicRBNOutputSchemes.LINRAND:
        G = rbnToNetX(RBN)
        btw = nx.betweenness_centrality(G)
        ps = array([btw[x] for x in btw.keys()])
        ps = ps / np.sum(ps)
        
    def invertProbs(p):
        p = 1 - p
        p = p - min(p)
        p = p / np.sum(p)        
        return p
    
    #external outputs    
    nExtOuts = int(N*outputSize)
    if externalOutputAllocScheme == monolithicRBNOutputSchemes.LINRAND:
        outputs = [x for x in range(N)]
        random.shuffle(outputs)
        RBN['outputNodes'] = outputs[:nExtOuts]
    else:
        outPs = copy.copy(ps)
        if  externalOutputAllocScheme == monolithicRBNOutputSchemes.BTWCENTRALITYLOW:
            outPs = invertProbs(outPs)
            
        outProbDist = rv_discrete(name='outProbDist', values=(list(btw.keys()), outPs))
        
        RBN['outputNodes'] = [0 for x in range(nExtOuts)]
        for i in range(nExtOuts):
            RBN['outputNodes'][i] = outProbDist.rvs()
            
        
        
    
    #external inputs
    nExtIns = int(externalInputConnectivity * N)
    if externalInputAllocScheme == monolithicRBNInputSchemes.LINRAND:
            nodesWithInCapacity = [x for x in range(RBN['N']) if len(RBN['net'][x]) < 4]
            for i in range(nExtIns):
                node = random.choice(nodesWithInCapacity)
                inputIdx = np.random.randint(N, N+nIns)
                RBN['net'][node].append(inputIdx)
                if len(RBN['net'][node]) == 4:
                    nodesWithInCapacity.remove(node)
                    
    else:
        inPs = copy.copy(ps)
        if  externalInputAllocScheme == monolithicRBNInputSchemes.BTWCENTRALITYLOW:
            inPs = invertProbs(inPs)
            
        inProbDist = rv_discrete(name='inProbDist', values=(list(btw.keys()), inPs))
        extInCount = nExtIns
        while extInCount > 0:
            node = inProbDist.rvs()
            if len(RBN['net'][node]) < 4:
                inputIdx = np.random.randint(N, N+nIns)
                RBN['net'][node].append(inputIdx)
                extInCount -= 1
        


    RBN['Kavg'] = inTotal/N
    return RBN

class modularRBNInputSchemes(Enum):
    LINRAND = 0
    BTWCENTRALITYHIGH = 1
    BTWCENTRALITYLOW = 2
    MODULE=3

class modularRBNOutputSchemes(Enum):
    LINRAND = 0
    BTWCENTRALITYHIGH = 1
    BTWCENTRALITYLOW = 2
    MODULE=3

def growModularRBN(M, n, m0, Q, targetN, externalInputConnectivity = 0.3, 
            externalInputAllocScheme = modularRBNInputSchemes.LINRAND,
                   nIns=1,
            externalOutputAllocScheme = modularRBNOutputSchemes.LINRAND,
                   outputSize=0.1):
    def testQInequality(qprobs):
        ok=True
        for i in range(len(qprobs)-1):
            termok = True
            term1 = 1 - qprobs[i+1]
            term2 = qprobs[i+1] * (1 - qprobs[i])
            if (term1 > term2):
                termok = False
            print(f"{term1} <= {term2} : {termok}")
            if not termok:
                ok = False
        return ok
    if not testQInequality(Q):
        print("WARNING: Q Inequality failed")
    RBN = {'p':0.5, 'nIns': 1, 'externalInputConnectivity':externalInputConnectivity}
    RBN['net'] = []
    RBN['module'] = []
    #initial modules
    nModules = pow(n, M-1)
    RBN['hier'] = [[x] for x in range(nModules)]
    N=m0 * nModules
    nGroups = int(nModules / n)
        
    
    outputs = [x for x in range(targetN)]
    random.shuffle(outputs)
    RBN['outputNodes'] = outputs[:int(targetN*outputSize)]
    
    
    def addNode(module, connections):
        RBN['net'].append(connections)
        RBN['module'].append(module)
        
    for i in range(M-1):
#         print(f"g:{nGroups}")
        for g in range(nGroups):
            groupSize = int(nModules/nGroups)
            for j in range(groupSize):
                RBN['hier'][(g*groupSize) + j].append(g)
        nGroups = int(nGroups / n)
        
    #add initial nodes
    for m in range(nModules):
        nodeIdxs = set([(m*m0) + x for x in range(m0)])
        for node in range(m0):
            connectedNodes = nodeIdxs - {(m*m0) + node}
            addNode(m, list(connectedNodes))
    
    def selectWithPrefAttachment(candidates):
        numConnections = [len(RBN['net'][x]) for x in candidates]
        numConnections = np.array(numConnections) / np.sum(numConnections)
        custm = rv_discrete(name='custm', values=(candidates, numConnections))
        chosenNode = custm.rvs()
        return chosenNode
    
    while N < targetN:
        #check the network isn't full
        if sum([len(x) for x in RBN['net']]) == N * 4:
            break
        opLevel = M
        while opLevel > 1:
#             print(f"Oplevel: {opLevel}")
            if (random.random() < Q[opLevel-2]):
                #in module connection
                if (opLevel > 2):
                    opLevel -= 1
#                     print("in-module - descending")
                else:
#                     print(f"adding node {N}")
                    targetModule = random.randint(nModules)
                    targetNodes = [x for x in range(N) if RBN['module'][x] == targetModule]
#                     print(targetModule)
#                     print(targetNodes)
                    if len(targetNodes) >= m0:
                        connectedNodes = []
                        for cx in range(m0):
                            targetNode = selectWithPrefAttachment(targetNodes)
                            connectedNodes.append(targetNode)
                            targetNodes.remove(targetNode)
                        addNode(targetModule, connectedNodes)
#                         print(f"adding node in module {targetModule}, cx: {connectedNodes}")
                        N += 1
                    else:
#                         print("no candidate nodes")
                        None
                    break

            else:
                #between module connection
                module1=1 
                module2=1
                sublevel = 0
                node1=node2=0
                makeAConnection = True
#                 if (opLevel > 1):
                sublevel = opLevel-2
                numModulesAtLevel = pow(n, M-sublevel-1)
                candidates = [x for x in range(numModulesAtLevel)]
                module1 = random.choice(candidates)
                candidates.remove(module1)
                module2 = random.choice(candidates)
#                     print(candidates, module1, module2, sublevel)
#                     print(N, len(RBN['net']),  len(RBN['module']))
                mods1 = [x[0] for x in RBN['hier'] if x[opLevel-2] == module1]
                mods2 = [x[0] for x in RBN['hier'] if x[opLevel-2] == module2]
#                     print(mods1)


                node1Candidates = [x for x in range(N) if RBN['module'][x] in mods1 and len(RBN['net'][x]) < 4]
#                     print(node1Candidates)
                node2Candidates = [x for x in range(N) if RBN['module'][x] in mods2]
#                     print(node2Candidates)
                if len(node1Candidates) > 0:
                    node1 = selectWithPrefAttachment(node1Candidates)
                    node2 = selectWithPrefAttachment(node2Candidates)
                else:
                    makeAConnection=False
#                 else:
                #bottom level, all nodes are candidates
                nodeCandidates = [x for x in range(N) if len(RBN['net'][x]) < 4]
                if (len(nodeCandidates) > 1):
                    node1 = selectWithPrefAttachment(nodeCandidates)

                    nodeCandidates = [x for x in range(N)] 
                    nodeCandidates.remove(node1)
                    node2 = selectWithPrefAttachment(nodeCandidates)
                else:
                    makeAConnection = False

                if makeAConnection:
                    RBN['net'][node1].append(node2)
#                     print(f"making between-module cx at level {opLevel} from ({module1},{node1}) to ({module2},{node2})")
                else:
                    None
#                     print("no candidates")
              
                break

    RBN['N'] = N
    RBN['nodes'] = [random.random() > RBN['p'] for x in range(N)] 
    RBN['lookups'] = [[random.random() > RBN['p'] for ltentry in range(16)] for node in range(N)]  
    RBN['binMult'] = [1,2,4,8]
    
    #Kavg
    RBN['Kavg'] = np.mean([len(x) for x in RBN['net']])

    G = btw = ps = None
    if externalInputAllocScheme != monolithicRBNInputSchemes.LINRAND or externalOutputAllocScheme != monolithicRBNOutputSchemes.LINRAND:
        G = rbnToNetX(RBN)
        btw = nx.betweenness_centrality(G)
        ps = array([btw[x] for x in btw.keys()])
        ps = ps / np.sum(ps)
        
    def invertProbs(p):
        p = 1 - p
        p = p - min(p)
        p = p / np.sum(p)        
        return p


    
    #external outputs
    nExtOuts = int(N*outputSize)
    if externalOutputAllocScheme == modularRBNOutputSchemes.LINRAND:
        outputs = [x for x in range(N)]
        random.shuffle(outputs)
        RBN['outputNodes'] = outputs[:nExtOuts]
    elif externalOutputAllocScheme == modularRBNOutputSchemes.MODULE:
        #assign outputs to the minimum number of modules
        allModules = arange(nModules)
        random.shuffle(allModules)
        extOutCount = nExtOuts
        currModuleNodes = []
        RBN['outputNodes'] = []
        while extOutCount > 0:
            if len(currModuleNodes)==0:
                currModule = allModules[0]
                allModules = allModules[1:]
                currModuleNodes = [i for i,x in enumerate(RBN['module']) if x == currModule]
                random.shuffle(currModuleNodes)
            currNode = currModuleNodes[0]
            currModuleNodes = currModuleNodes[1:]
            RBN['outputNodes'].append(currNode)
            extOutCount -= 1
            

    else:
        outPs = copy.copy(ps)
        if  externalOutputAllocScheme == modularRBNOutputSchemes.BTWCENTRALITYLOW:
            outPs = invertProbs(outPs)
            
        outProbDist = rv_discrete(name='outProbDist', values=(list(btw.keys()), outPs))
        
        RBN['outputNodes'] = [0 for x in range(nExtOuts)]
        for i in range(nExtOuts):
            RBN['outputNodes'][i] = outProbDist.rvs()
                

    #external inputs
    nExtIns = int(externalInputConnectivity * N)
    if externalInputAllocScheme == modularRBNInputSchemes.LINRAND:
            nodesWithInCapacity = [x for x in range(RBN['N']) if len(RBN['net'][x]) < 4]
            for i in range(nExtIns):
                node = random.choice(nodesWithInCapacity)
                inputIdx = np.random.randint(N, N+nIns)
                RBN['net'][node].append(inputIdx)
                if len(RBN['net'][node]) == 4:
                    nodesWithInCapacity.remove(node)
                    
    elif externalInputAllocScheme == modularRBNInputSchemes.MODULE:
        #assign inputs to the minimum number of modules
        allModules = arange(nModules)
        random.shuffle(allModules)
        extInCount = nExtIns
        currModuleNodes = []
        while extInCount > 0:
            if len(currModuleNodes)==0:
                currModule = allModules[0]
                allModules = allModules[1:]
                currModuleNodes = [i for i,x in enumerate(RBN['module']) if x == currModule]
                random.shuffle(currModuleNodes)
            currNode = currModuleNodes[0]
            currModuleNodes = currModuleNodes[1:]
            if len(RBN['net'][currNode]) < 4:
                inputIdx = np.random.randint(N, N+nIns)
                RBN['net'][currNode].append(inputIdx)
                extInCount -= 1
        
    else:
        inPs = copy.copy(ps)
        if  externalInputAllocScheme == modularRBNInputSchemes.BTWCENTRALITYLOW:
            inPs = invertProbs(inPs)
            
        inProbDist = rv_discrete(name='inProbDist', values=(list(btw.keys()), inPs))
        extInCount = nExtIns
        while extInCount > 0:
            node = inProbDist.rvs()
            if len(RBN['net'][node]) < 4:
                inputIdx = np.random.randint(N, N+nIns)
                RBN['net'][node].append(inputIdx)
                extInCount -= 1

    return RBN







def oneHot(v):
    outs = zeros(numClasses, dtype=int)
    outs[v] = 1
    return list(outs)

def evalExample(targetOutput, netOut):
    #return score and array indicating which output are correct, bool - if to include in training batch
    posReinforce = list(ones(len(netOut), dtype=int))
    negReinforce = list(zeros(len(netOut), dtype=int))
#     expectedOutput = oneHot(targetOutput)
    expectedOutput = [targetOutput]
#     testResults = [1 if netOut[i] == x else 0 for i,x in enumerate(expectedOutput)]
    #                 return sum(testResults)
#     score = 1 if sum(testResults) == len(testResults) else 0
    score = 1 if expectedOutput == netOut else 0
#     nodeCorrections = posReinforce if score == 1 else negReinforce
    nodeCorrections = [1 if netOut[i] == x else 0 for i,x in enumerate(expectedOutput)]
    numCorrect = sum(nodeCorrections)
    trainOnThis = (numCorrect == len(netOut)) or (numCorrect == 0)# or (random.random() < 0.1)

    return [score, nodeCorrections, trainOnThis]


def selectTrainingSet(evalResults):
    psves = [x for x in evalResults if x[0] == 1]
    negs = [x for x in evalResults if x[0] == 0]
    smallest = min (len(psves), len(negs))
    psveProb = 0 if len(psves)== 0 else smallest / len(psves)
    negProb = 0 if len(negs) == 0 else smallest / len(negs)
#     evalResults[0][2] == False
    for i,x in enumerate(evalResults):
        if (evalResults[i][0] == 0):
            evalResults[i][2] = random.random() < negProb
        else:
            evalResults[i][2] = random.random() < psveProb
#     print("sel ", negProb, ", ", psveProb)
    return evalResults

    
    
    
def createTrainingSet(allDataIn, allDataOut, testSize=0.2):
#     allDataIn =  states
#     allDataOut = array(outputPDM, dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(allDataIn, allDataOut, test_size=testSize)
    return {'in':x_train, 'out':y_train, 
            'testin': x_test, 'testout': y_test, 
            'eval':evalExample,
            'selector': selectTrainingSet
           }


wavelen = 16
phasor = [(v %wavelen)/wavelen for v in arange(wavelen * 4)]
w = [(sin(v * math.pi *2)+1) /2 for v in phasor]
w2 = [1-v for v in phasor]

PDMoversample = 2
wpdm = pdm(w,PDMoversample)
wpdm2 = pdm(w2,PDMoversample)
phasorpdm = pdm(phasor,PDMoversample)

byteToBits4Table = [byteToBits(x,4) for x in range(16)]
phasorPE = [byteToBits(int(x * 255.0), 8) for x in phasor]

inputPDM = wpdm[0][:-1]
outputPDM = wpdm[0][1:]


def getRBNStates(rbnModel, washout=0):
    oversample = 1
    inputPE = [[x] for x in inputPDM] #vstack(array(inputPDM, dtype=int))
    steps = len(inputPE)
    stateSize = len(rbnModel['outputNodes']) #int(rbnModel['N'] * trainingFraction)
    states = [[False for x in range(stateSize)] for y in range(steps*oversample)]
    inputs = [[False for x in range(rbnModel['nIns'])] for y in range(steps*oversample)]
    inputNodes = [0 for x in range(rbnModel['nIns'])]
    rbnIdx=0
    for step in range(washout):
        inputNodes = inputPE[rbnIdx % len(inputPE)]
#         print(inputNodes)
        rbnModel = runRC(rbnModel,  inputNodes)
        rbnIdx += 1
    rbnIdx=0
    for step in range(steps):
        inputNodes = inputPE[rbnIdx % len(inputPE)]
        for overstep in range(oversample):
            rbnModel = runRC(rbnModel,  inputNodes)
            if (step==0 and overstep==0):
                initState = rbnModel['nodes'].copy()
            states[(step*oversample)+overstep] = [rbnModel['nodes'][x] for x in rbnModel['outputNodes']]
            inputs[(step*oversample)+overstep] = inputNodes
        rbnIdx += 1
    return inputs,states, initState

from enum import Enum
class initSchemes(Enum):
    LINRAND= 0
    ZEROS = 1
    SPARSE = 2
    
def createNet(inSize, layerSizes, initScheme = initSchemes.LINRAND, lowProbLayers=[]):
    testNet = lutnet.FFLUT4Net(inSize)
    for v in layerSizes:
        testNet.addLayer(v)
    if (initScheme != initSchemes.LINRAND):
        #leave the top layer as linrand
        for i_layer in range(testNet.getLayerCount()-1):
            if i_layer not in lowProbLayers:
#                 print (i_layer)
    #             print(i_layer, end=": ")
                for i_node in range(testNet.getLayerSize(i_layer)):
    #                 print(i_node, end=",")
                    for i_ttable in range(16):
                        testNet.setTtableElement(i_layer, i_node, i_ttable, 0)
                    if (initScheme == initSchemes.SPARSE):
                        testNet.setTtableElement(i_layer, i_node, np.random.randint(16), 1)
    #                 for i_ttable in range(16):
    #                     print(testNet.getTtableElement(i_layer, i_node, i_ttable), end="")

    #                 print(" ",end="")
    #             print("")        
        
    return testNet

#need to make a map of interlayer dependencies after creating the network
import copy
def findLayerClusters(lutnet, layer, nodes):
    clusterList = []
    clusterTemplate = {'outputNodes':[], 'inputNodes':[]}
    def makeNewCluster(topNode):
        newCluster = copy.deepcopy(clusterTemplate)
        newCluster['outputNodes'] = [topNode]
        for i_input in range(4):
            newCluster['inputNodes'].append(lutnet.getInputIndex(layer, topNode,i_input))
        return newCluster
        
    for i_toplayer in nodes:
        if len(clusterList) == 0:
            clusterList = [makeNewCluster(i_toplayer)]
        else:
            foundCluster = None
            for i_input in range(4):
                inNodeIdx = lutnet.getInputIndex(layer, i_toplayer, i_input)
                for cluster in clusterList:
                    if inNodeIdx in cluster['inputNodes']:
                        foundCluster = cluster
                        break
                if foundCluster != None:
                    break
            if foundCluster != None:
                #add to current cluster
                cluster['outputNodes'].append(i_toplayer)
                for i_input in range(4):
                    inputNodeIdx = lutnet.getInputIndex(layer, i_toplayer,i_input)
                    if inputNodeIdx not in cluster['inputNodes']:
                        cluster['inputNodes'].append(inputNodeIdx)
            else:
                #make a new cluster
                clusterList.append(makeNewCluster(i_toplayer))
#         print('========', i_toplayer)
#         print(clusterList)
            

    return clusterList

def splitIntoBatches(x, batchSize):
    rem = x.shape[0] % batchSize
    dim1 = x.shape[0]-rem
    batches1 = list(x[:dim1].reshape(int(dim1/batchSize),batchSize))
    batches2 = x[-rem:]
    batches1[len(batches1)-1] = np.concatenate((batches1[len(batches1)-1], batches2))
    return batches1


def genStruct(depth, numclasses, topLayerIndependence=2):
    st = [numclasses]
    v = numclasses
    print(st)
    for i in range(depth):
        v *= 4 if i < topLayerIndependence else 2
        st.append(v)
    st.reverse()
    return st


#this is the new version that trains by layer instead of node
#inputs: network, training pair
#output: list of ttable entries to change and testresult for this training pair
def trainingStep(lutnet, testResults, config, layerDepth):
    
    def processLayer(layerIdx, incorrectNodes):
#         print(f'Process layer {layerIdx} with incorrect nodes {incorrectNodes}')
        #divide into dependency groups
        #TODO: to optimise, cache these cluster results
        layerClusters = findLayerClusters(lutnet, layerIdx, incorrectNodes)
#         print(incorrectNodes)
#         print(layerClusters)
        
        nodeTtableChanges = []
        nodeTtableProtections = []
        
        stats = {}
        stats['comboMissCount'] = 0
        stats['noCompatCount'] = 0
        stats['ruleAddition'] =0

        

        for cl in layerClusters:
#             print("Processing cluster: ", cl)
            #find ttable options for each node and check for compatible options
            #if there are compatible options, then choose the one that needs minimum change
            #if there are no compatible options, do something else, but what? (definitely do something or risk stalling)
            changeCandidateLists = {}
            for nodeIdx in cl['outputNodes']:
                ttable = [lutnet.getTtableElement(layerIdx, nodeIdx, x) for x in range(16)]
                #find ttable entries that would be correct, if given correct input
#                 targetResult = data['out'][trainingSetIndex][nodeIdx]
                targetResult = lutnet.getOutput(layerIdx,nodeIdx)
                if (nodeIdx in incorrectNodes):
                    targetResult = 1 - targetResult
                
                LUTChangeCandidates = [i for i,x in enumerate(ttable) if x == targetResult]
#                 print("Change candidates for node ", nodeIdx, ": ", LUTChangeCandidates)
                
                #policy
#                 if len(LUTChangeCandidates) == 0 and config['MAKE_RULE_ADDITIONS']==True:
#                     #choose a random candidate
#                     newCandidate = np.random.randint(16)
#                     #set a rule for it
#                     lutnet.setTtableElement(layerIdx, nodeIdx, newCandidate, targetResult)
#                     #add it to the list
#                     LUTChangeCandidates = [newCandidate]
#                     stats['ruleAddition'] += 1

#                     print(f"Policy: created new rule for layer {layerIdx} node {nodeIdx}")
                changeCandidateLists[nodeIdx] = LUTChangeCandidates
#             print("All change candidates: ", changeCandidateLists)
            #for all poss combos of candidates
            changeCombos = [changeCandidateLists[x] for x in changeCandidateLists]
            for v in changeCombos:
                random.shuffle(v)

            inputMappings = [[lutnet.getInputIndex(layerIdx, nodeIdx,x) for x in range(4)] for nodeIdx in cl['outputNodes']]
#             print("inputmappings: ", inputMappings)
            compatibleChangeCandidates = []
            i_combo=0
#             comboSize = sum([len(changeCandidateLists[x]) for x in changeCandidateLists])
#             if(comboSize < 50):
            for combo in itertools.product(*changeCombos):
#                 print(i_combo, end='')
                i_combo+=1
                #stack up the bits
                #do they all match? if so, add this combo to list (change candidates and bit pattern)
#                 print("combo: ", combo)
                bitstacks = {}
                for i_node,v in enumerate(combo):
                    ttableEntryBits = byteToBits4(v)
#                     print(ttableEntryBits)
#                     print('input mappings for node ', cl['outputNodes'][i_node] ,": ", inputMappings[i_node])
                    for i_bit, ttableBit in enumerate(ttableEntryBits):
                        bitSource = inputMappings[i_node][i_bit]
                        if bitSource in bitstacks:
                            bitstacks[bitSource] += ttableBit
                        else:
                            bitstacks[bitSource] = ttableBit

#                 print("bitstacks: ", bitstacks)
                #check for compatibility
                compatible = True
                numNodes = len(cl['outputNodes'])
                #was the bitstack either all zeros or all ones?
                for v in bitstacks:
#                     print(v)
                    if not (bitstacks[v] ==0 or bitstacks[v] == numNodes):
                        compatible = False
                        break
#                 print("Compatible: ", compatible)
                bitpattern = [1 if bitstacks[x] > 0 else 0 for x in bitstacks]
                if compatible:
                    compatibleChangeCandidates.append([combo, bitpattern, bitstacks])
                if (i_combo >= config['COMBO_LIMIT']):
#                     print("CL", end="")
                    stats['comboMissCount'] += 1
                    break

#             print("compatibleChangeCandidates: ", compatibleChangeCandidates)
            if len(compatibleChangeCandidates) > 0:
                #input state for all nodes in the cluster
                inputState = [lutnet.getOutput(layerIdx-1,x) for x in cl['inputNodes']]
#                 print(cl)
#                 print(layerIdx)
#                 print(inputState)
#                 print(compatibleChangeCandidates)
                distancesToChangeCandidates = array([distance.hamming(inputState, x[1]) for x in compatibleChangeCandidates])
                minDist = min(distancesToChangeCandidates)
                minDistIdxs = [i for i,x in enumerate(distancesToChangeCandidates) if x <= minDist]
                indexChoice = np.random.choice(minDistIdxs)
                comboChoice = compatibleChangeCandidates[indexChoice]
#                 print("Input: ", inputState, ", modification: ", comboChoice)
                #which nodes should be changed base on current rule of input nodes?
                for i_innode, v in enumerate(comboChoice[2]):
                    curInputVal = lutnet.getOutput(layerIdx-1,v)
                    targetInputVal = 1 if comboChoice[2][v] > 0 else 0
#                     print(v, targetInputVal, curInputVal)
                    nodeInputIndexes =[lutnet.getInputIndex(layerIdx-1, v, x) for x in range(4)]
#                         print("Node input idxs: ", nodeInputIndexes)
                    nodeInputState = [lutnet.getOutput(layerIdx-2,x) for x in nodeInputIndexes]
                    rule = bitsToInt(nodeInputState)
                    if (config['RANDOM_VAL_CHOICE']==True):
                        rule = np.random.randint(16)
                        targetInputVal = np.random.randint(2)
                    if targetInputVal != curInputVal:
                        nodeTtableChanges.append({'layer': layerIdx-1, 'nodeIdx':v, 'rule':rule, 'val':targetInputVal})
#                         print(nodeTtableChanges)
                    else:
                        nodeTtableProtections.append({'layer': layerIdx-1, 'nodeIdx':v, 'rule':rule, 'val':targetInputVal})
#                 print("input mappings: ", inputMappings)
        
#         print("changes: ", nodeTtableChanges)
        return nodeTtableChanges, nodeTtableProtections, stats
        
    nodeChanges = []
    nodeProtections = []
#     state = data['in'][trainingSetIndex]
#     lutnet.calc(state)
#     testResults = [1 if lutnet.getTopLayerOutput(i) == x else 0 for i,x in enumerate(data['out'][trainingSetIndex])]
    changeList = []
    topLayer = lutnet.getLayerCount() -1
    incorrectNodes = [i for i,x in enumerate(testResults) if x == 0]
    stats = {}
#     print("Incorrect nodes in top layer: ", incorrectNodes)
    #choose how deep to go in node changes
    for layer in range(layerDepth):
#             print(f"Processing layer {layer+1}/{layerDepth}")
#             if len(incorrectNodes) > 0:
        nodeChanges, nodeProtections, layerStats = processLayer(topLayer - layer, incorrectNodes)

        incorrectNodes = np.unique([x['nodeIdx'] for x in nodeChanges])

        for key in layerStats:
            if not key in stats:
                stats[key] = 0
            stats[key] += layerStats[key]

    
    return nodeChanges, nodeProtections, stats

import copy
class reRandSchemes(Enum):
    NONE=0
    TOPLAYER=1
    BOTTOMLAYER=2

class layerChoice(Enum):
    RANDOM_LOG=0
    CYCLIC=1
    RANDOM_LIN=2

class layerChoicePoint(Enum):
    DATASET=0
    BATCH=1

def trainNetwork(iterations, ts, batchErrorLog, config, layerLog, modelIdx):
    def splitIntoBatches(x, batchSize):
        rem = x.shape[0] % batchSize
        dim1 = x.shape[0]-rem
        batches1 = list(x[:dim1].reshape(int(dim1/batchSize),batchSize))
        batches2 = x[-rem:]
        batches1[len(batches1)-1] = np.concatenate((batches1[len(batches1)-1], batches2))
        return batches1
    minerror = 999
    visDebug = 0
    if config['STARTING_MODEL'] == None:
        net = createNet(len(ts['in'][0]), config['LAYERS'], config['INITSCHEME'], [])
    else:
        net = config['STARTING_MODEL']
    trainSize=len(ts['in'])
    trainLength = iterations

    if visDebug:
        figsize(20,50)
        fig, axes = plt.subplots(nrows=trainLength, ncols=4)
        ax = axes.flatten()

    idx=0
    graphIdx=0
    batchSize = config['BATCHSIZE']
    outputLayerSize = net.getLayerSize(net.getLayerCount()-1)
    minError = 1
    lastError = 1.0
    finished=0
    shuffledIndexes = arange(len(ts['in']))
    np.random.shuffle(shuffledIndexes)
#     batches = shuffledIndexes.reshape((int(len(shuffledIndexes)/batchSize),batchSize))
    batches = splitIntoBatches(shuffledIndexes, batchSize)
    changeListLengths = []
    noiseProb = 0
    lastNoise=0
    bestModel = 0
    lastModel = 0
    trainStats = []
    def chooseLayerDepth():
        layerIndexes = arange(net.getLayerCount()-2) + 1
        layerIndexes = np.setdiff1d(layerIndexes, config['FIXED_LAYERS'])

#         print('layerIndexes: ', layerIndexes)
            
        if (config['LAYER_CHOICE'] == layerChoice.CYCLIC):
            layerDepth = layerIndexes[i % len(layerIndexes)]
        else:
            if (config['LAYER_CHOICE'] == layerChoice.RANDOM_LIN):
                layerDepth = np.random.choice(layerIndexes)
            else:
                #RANDOM_LOG
                layerProbs = layerIndexes / sum(layerIndexes)
                layerDepth = np.random.choice(layerIndexes,1,p=layerProbs)[0]
        return layerDepth
    for i in range(trainLength):
#     for i in notebook.tnrange(trainLength):            
        trainingScore =0.0
        layerProbCurve=1.0
        if (i%2==0):
            print(f'model {modelIdx}, i {i}, {minError}')

        if (config['RESHUFFLE_BATCHES'] == True):
            np.random.shuffle(shuffledIndexes)
            batches = splitIntoBatches(shuffledIndexes, batchSize)
            
#         print(i, "layer depth", layerIndexes, layerDepth)
        if (config['LAYER_CHOICE_POINT'] == layerChoicePoint.DATASET):
            layerDepth = chooseLayerDepth()
        stepStats = {}
        stopCount = 0
        for batch in batches:
#             print(f"Batch: {batch}, len({len(batch)})")
            logging.debug("Batch %s", batch)
            if (config['LAYER_CHOICE_POINT'] == layerChoicePoint.BATCH):
                layerDepth = chooseLayerDepth()
            changeListAll = []
            keepListAll = []
            batchError = 0
            

            def calcNet(trainIndex):
                state = ts['in'][trainIndex]
                net.calc(state)
                netOutput = [net.getTopLayerOutput(i) for i in range(outputLayerSize)]
                return netOutput

            def trainOneExample(trainIndex, nodeCorrections):
                netOutput = calcNet(trainIndex)
                changeList, keepList, exampleStats = trainingStep(net, nodeCorrections, config, layerDepth)
#                 print('clistlen: ',len(changeList))
                for key in exampleStats:
                    if not key in stepStats:
                        stepStats[key] = 0
                    stepStats[key] += exampleStats[key]

                return array(nodeCorrections), changeList, keepList
            
                
            batchOutputs = [calcNet(x) for x in batch]
#             print('outputs ', batchOutputs)
#             score, nodeCorrections, True
            batchEvals = [ts['eval'](ts['out'][x], batchOutputs[i]) for i,x in enumerate(batch)]
#             print('evals', batchEvals)
            if config['SELECTOR']:
                batchEvals = ts['selector'](batchEvals)
            batchChanges = [trainOneExample(x, batchEvals[i][1]) for i,x in enumerate(batch) if batchEvals[i][2] == True]

            numChanges=0
            avgTestRes = 0
            for res in batchChanges:
                testRes, changeList, keepList = res
#                 print('batchres: ',testRes, changeList)
                changeListAll.append(changeList)
                keepListAll.append(keepList)
            avgTestRes = np.mean([x[0] for x in batchEvals]) 
            batchError = 1.0 - avgTestRes
            lastError = batchError
            batchErrorLog.append(batchError)

            if (batchError <= minError):
                minError = batchError
                print("m{} {}\tmin:{}".format(modelIdx, i, minError))
                bestModel = pickle.dumps(net)
            
            if (batchError<= config['STOP_CONDITION']):
                stopCount += 1
                if stopCount == len(batches):
                    finished=1
                    break;
            else:
                stopCount=0
#                 print("processing changes")
#                 print('clistalllen',len(changeListAll))
#                 print(changeListAll)
                if len(changeListAll) > 0:
                    #apply batch changes
#                     random.shuffle(changeListAll)
                    changeFreqTable = {}
                    def makeHash(layer, node, rule):
                        # [16 bit layer][32 bit node][4 bit rule]
                        layer = (layer & 65535) << 36
                        node = ((node &  4294967295) << 4)
                        rule = (rule & 15)
                        return layer | node | rule
                    def hashToAddress(hash):
                        return [hash >> 36, (hash >> 4) & 4294967295, hash & 15]
#                     print(".",end='')
                    if (config['POPULARITY'] > 0):
                        for changes in changeListAll:
                            for c in changes:
                                hash = makeHash(c['layer'], c['nodeIdx'], c['rule'])
                                if hash not in changeFreqTable:
    #                                 print("adding entry for ", c)
                                    changeFreqTable[hash] = [0,0]
                                changeFreqTable[hash][c['val']]+=1
    #                     print(changeListAll)
    #                     print(changeFreqTable)
                        if (config['KEEP_LIST'] == True):
                            for changes in keepListAll:
                                for c in changes:
                                    hash = makeHash(c['layer'], c['nodeIdx'], c['rule'])
                                    if hash not in changeFreqTable:
        #                                 print("adding entry for ", c)
                                        changeFreqTable[hash] = [0,0]
    #                                 else:
    #                                     print(".")
                                    changeFreqTable[hash][c['val']]+=1
#                                     print("reduce ", changeFreqTable[hash])
    #                                 if (changeFreqTable[hash][c['val']] > 0):
    #                                     changeFreqTable[hash][c['val']] = 0

    #                     print(changeListAll)
#                         print(changeFreqTable)

    #                     print(";",end='')
                        changeMags = [] #zeros(len(changeFreqTable))
                        #todo: need sorted here? probably not
                        for i_hash,hash in enumerate(changeFreqTable):
#                             address = hashToAddress(hash)
                            changeMag = max(changeFreqTable[hash])
                            if (changeMag > 0):
                                changeMags.append(changeMag)
#                             print(address, probs, val)
#                         print(changeMags)
                        if (len(changeMags) > 0):
                            changeThresh = np.quantile(changeMags, config['POPULARITY'])
                            for hash in changeFreqTable:
                                changeVal = changeFreqTable[hash]
                                mag = max(changeVal)
                                if mag >= changeThresh:
                                    address = hashToAddress(hash)
                                    val = 0 if changeVal[0] > changeVal[1] else 1
                                    net.setTtableElement(address[0], address[1], address[2], val)
#                                     print(address)
                    else:
                        for changes in changeListAll:
                            for c in changes:
                                net.setTtableElement(c['layer'], c['nodeIdx'], c['rule'],c['val'])
    #                             print(f"{c['layer']}, {c['nodeIdx']}, {c['rule']},{c['val']}")
    

                            
#             layerLog.append(net.serialiseAllLayers())
        if (stepStats != {}):
            trainStats.append(stepStats)    
        if finished:
            break
    lastModel = pickle.dumps(net)
    return bestModel, lastModel, trainStats
            


    

import random as pyrand
def testOnData(net, dataIn, dataOut, evalFunction):
    score=0
    outputLayerSize=net.getLayerSize(net.getLayerCount()-1)
    for i,t in enumerate(dataIn):
        net.calc(t)
        result = [net.getTopLayerOutput(x) for x in range(outputLayerSize)]
        itemScore = evalFunction(dataOut[i], result)[0]
        score += itemScore
    score = score/len(dataIn) * 100
    return score

structs = [
#         [1024,256,64,16,4,1],
            [4096,1024,256,64,16,4,1],
]

class rbnTypes(Enum):
    MONOLITHIC=0
    MODULAR=1
    


import random as pyrand
def testOnData(net, dataIn, dataOut, evalFunction):
    score=0
    outputLayerSize=net.getLayerSize(net.getLayerCount()-1)
#     batchOutputs = [net.calc(x) for x in dataIn]
#     batchEvals = [trainingSet['eval'](trainingSet, i, batchOutputs[i]) for i in range(len(dataIn))]
#     score = np.mean([x[0] for x in batchEvals]) * 100
    for i,t in enumerate(dataIn):
        net.calc(t)
        result = [net.getTopLayerOutput(x) for x in range(outputLayerSize)]
        itemScore = evalFunction(dataOut[i], result)[0]
        score += itemScore
    score = score/len(dataIn) * 100
    return score

#[struct, fixed layers (depths)] 
structs = [
#     [512,128,32,8],
#     [128,32,8,4],
#         [1024,256,64,16,4,1],
#         [256,64,16,4,1],
    

            [4096,1024,256,64,16,4,1],
#     [512,256,64,16,4],
#     [512,128,32,8],
#     [256,128,32,8,4],
#     [512,128,32,8,2],
#     [2048,512,128,32,8],
#     [2048,512,128,128,32,8]
]


    


@ray.remote    
def trainANetwork(i_test):
    starttime = datetime.now().timestamp()
    structIdx = np.random.randint(len(structs))
    depth = np.random.randint(3,5)
    ind = np.random.randint(0,1)# max(1,depth - random.randint(0,depth))
    rbnConfigs = [
        {'rbnType':rbnTypes.MONOLITHIC, 'extInputAlloc':monolithicRBNInputSchemes.LINRAND, 'extOutputAlloc':monolithicRBNOutputSchemes.LINRAND},
        {'rbnType':rbnTypes.MONOLITHIC, 'extInputAlloc':monolithicRBNInputSchemes.BTWCENTRALITYHIGH, 'extOutputAlloc':monolithicRBNOutputSchemes.BTWCENTRALITYHIGH},
        {'rbnType':rbnTypes.MONOLITHIC, 'extInputAlloc':monolithicRBNInputSchemes.BTWCENTRALITYLOW, 'extOutputAlloc':monolithicRBNOutputSchemes.BTWCENTRALITYHIGH},
        {'rbnType':rbnTypes.MONOLITHIC, 'extInputAlloc':monolithicRBNInputSchemes.BTWCENTRALITYHIGH, 'extOutputAlloc':monolithicRBNOutputSchemes.BTWCENTRALITYLOW},
        {'rbnType':rbnTypes.MONOLITHIC, 'extInputAlloc':monolithicRBNInputSchemes.BTWCENTRALITYLOW, 'extOutputAlloc':monolithicRBNOutputSchemes.BTWCENTRALITYLOW},

        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.LINRAND, 'extOutputAlloc':monolithicRBNOutputSchemes.LINRAND},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.BTWCENTRALITYHIGH, 'extOutputAlloc':modularRBNOutputSchemes.BTWCENTRALITYHIGH},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.BTWCENTRALITYHIGH, 'extOutputAlloc':modularRBNOutputSchemes.BTWCENTRALITYLOW},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.BTWCENTRALITYHIGH, 'extOutputAlloc':modularRBNOutputSchemes.MODULE},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.BTWCENTRALITYLOW, 'extOutputAlloc':modularRBNOutputSchemes.BTWCENTRALITYHIGH},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.BTWCENTRALITYLOW, 'extOutputAlloc':modularRBNOutputSchemes.BTWCENTRALITYLOW},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.BTWCENTRALITYLOW, 'extOutputAlloc':modularRBNOutputSchemes.MODULE},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.MODULE, 'extOutputAlloc':modularRBNOutputSchemes.BTWCENTRALITYHIGH},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.MODULE, 'extOutputAlloc':modularRBNOutputSchemes.BTWCENTRALITYLOW},
        {'rbnType':rbnTypes.MODULAR, 'extInputAlloc':modularRBNInputSchemes.MODULE, 'extOutputAlloc':modularRBNOutputSchemes.MODULE},
                 ]
    rbnConfigIndex = random.randint(len(rbnConfigs))
    rbnConfig =  rbnConfigs[rbnConfigIndex]
    testConfig={'LAYERCHOICE':'LOG', #or LOG 
                'LAYERPROBCURVE':1.0, 
                'BATCHSIZE':random.choice([32]),
                'STRUCT':structIdx,
                'LAYERS':structs[structIdx],
                'FIXED_LAYERS':[],
                'DEPTH':depth,
                'IND':ind,
                'LAYER_CHOICE_POINT': layerChoicePoint.BATCH,
                'RESHUFFLE_BATCHES': True,
                 'INITSCHEME': initSchemes.SPARSE,
#                 'STARTING_MODEL':savedModel #or None
                'STARTING_MODEL':None,
                'STOP_CONDITION': 0.00,
                'COMBO_LIMIT': 99999,
                'LAYER_CHOICE': layerChoice.RANDOM_LOG,
#                 'LAYER_CHOICE': np.random.choice([layerChoice.CYCLIC,layerChoice.RANDOM_LIN, layerChoice.RANDOM_LOG]),
                'RANDOM_VAL_CHOICE':False,
                'POPULARITY':0.90,
                'KEEP_LIST':True,
                'SELECTOR': False, #random.choice([False, True]),
                'RBNCONFIGIDX': rbnConfigIndex,
                'RBNCONFIG': rbnConfig
               }   
    N=500
    extInSize = 0.3
    extOutSize = 0.15
    if rbnConfig['rbnType'] == rbnTypes.MONOLITHIC:
        prob2 = 0.9 + (random.random() * 0.03)
        prob3 = pow(random.random(),0.5) * (0.945 - prob2)
        prob4 = 0.95 -prob2-prob3
        RBNm = makeRBN(N, [0.05,prob2, prob3, prob4], extInSize, externalInputAllocScheme = rbnConfig['extInputAlloc'], nIns=1, externalOutputAllocScheme = rbnConfig['extOutputAlloc'], outputSize=extOutSize)
    else:
        RBNm = growModularRBN(4,4,2,[0.85,0.9,0.94], N, extInSize, externalInputAllocScheme = rbnConfig['extInputAlloc'], nIns=1, externalOutputAllocScheme = rbnConfig['extOutputAlloc'], outputSize=extOutSize)
        
    rbnInputs, rbnStates, rbnInitState = getRBNStates(RBNm, PDMoversample * wavelen * 16)
    ts = createTrainingSet(rbnStates, array(outputPDM, dtype=int), testSize=0.1 )

    logging.info("config %s", testConfig)
    print("=======Test: ",i_test, "config: ", testConfig)
    errorLog = []
    layerLog = []
    bestModelPickled, lastModelPickled, trainStats = trainNetwork(600, ts, errorLog, testConfig, layerLog, i_test)
#     print(trainStats)
    endtime = datetime.now().timestamp()
    print("=======Test complete: ",i_test, "config: ", testConfig)
    print('log: ', min(errorLog), ', ', len(errorLog))
    def testModel(mod, trainSet):
        trainscore = testOnData(mod, trainSet['in'], trainSet['out'], trainSet['eval'])
        print(i_test, " Training Data Score:",trainscore, "%")
        testscore = testOnData(mod, trainSet['testin'], trainSet['testout'], trainSet['eval'])
        print(i_test, " Test Data Score:",testscore, "%")
        return trainscore, testscore
    print("Best model results:")
    bestModel = pickle.loads(bestModelPickled)
    trainscore, testscore = testModel(bestModel, ts)
#     print("Final model results:")
#     lastModel = pickle.loads(lastModelPickled)
#     lmtrainscore, lmtestscore = testModel(lastModel)
    result ={'layerLog': layerLog, 'modelPickled': bestModelPickled, 
#              'finalModelPickled': lastModelPickled, 
             'log':errorLog, 
#              'stats':trainStats,
             'cfg':testConfig, 
             'trainDataScore':trainscore, 'testDataScore':testscore, 
#              'lastModelTrainDataScore':lmtrainscore, 'lastModelTestDataScore':lmtestscore, 
             'time':endtime - starttime,
            'rbn':RBNm,
             'rbnCueState':rbnInitState,
            'ts':ts}
    return result
    
booscResults = []
tres = []
testCount=250
parallelCount = 100

from datetime import datetime
exptTS = datetime.now().strftime("%m%d%Y_%H%M%S")
filenameRoot = f"expt4-2Res/{sys.argv[0]}{exptTS}"
for i_test in range(testCount):
    filename = f"{filenameRoot}_t{i_test}_booscresults.picked"
    futures = [trainANetwork.remote(x) for x in range(parallelCount)]
    tres = ray.get(futures)
    for t in tres:
        t['ts']['eval'] = None
        t['ts']['selector'] = None
    with open(filename, 'wb') as handle:
        pickle.dump(tres,handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"===========================subexpt {i_test} complete")
    
print("Done - Fini - Klar - Faerdig - -Valmis - Completo")
