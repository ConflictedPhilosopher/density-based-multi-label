# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:11:01 2019

@author: xyan
"""

import numpy as np
import math
import pandas as pd
import numpy.matlib as b
from sklearn.preprocessing import normalize
import time
from entropy_estimators import *
import os.path

start = time.time()

def Input():
    # Read the data from the txt file
    DATA_HEADER = "flags"
    VALID_DATA_HEADER = DATA_HEADER + "-test"
    TRAIN_DATA_HEADER = DATA_HEADER + "-train"
    DATA_FOLDER = os.curdir
    NO_ATTRIBUTES = 19

    trainDataCSV = os.path.join(DATA_FOLDER, TRAIN_DATA_HEADER + "-csv.csv")
    validDataCSV = os.path.join(DATA_FOLDER, VALID_DATA_HEADER + "-csv.csv")
    completeDataFileName = os.path.join(DATA_FOLDER, DATA_HEADER + "-csv.csv")

    df = pd.read_csv(completeDataFileName)
    # df.drop(df.columns[0], axis=1, inplace=True)

    ML = []  # list of the multi-labels as a string
    for idx, row in df.iterrows():
        labels = [int(l) for l in row[NO_ATTRIBUTES:]]
        newlabel = "".join(map(str, labels))
        ML.append(newlabel)
    data = df.copy()
    data['multilabel'] = ML

    X = data.iloc[:, :NO_ATTRIBUTES]
    Y = data.iloc[:, NO_ATTRIBUTES:-1]

    label = Y
    return label


def Feature_Dist(label,Nc):
    Dist = []
    DisC,Dist = Com_Cal(label)
    

    return DisC, Dist

"""
Function Com_Cal

Description: Write a similarity calculation function to compute the correlation among
class labels
Input:
labels: class labelsets n-by-c matrix (n: number of samples; c: number of classes)
i: the ith class
j: the jth class
Output: a c-by-c similarity matrix
"""
def Com_Cal(labels):
    method = 'cosine'
    label_count = len(labels.columns)
    C = np.zeros((label_count, label_count))
    D = []
    if method == 'cosine':
        for i in range(label_count):
            for j in range(i, label_count):
                C[i, j] = np.dot(labels.iloc[:, i], labels.iloc[:, j])/ (np.linalg.norm(labels.iloc[:, i]) * np.linalg.norm(labels.iloc[:, j]))
                C[j, i] = C[i, j]
                D.append(C[i,j])
    return C,D

def fitness_cal(DisC,Nc,label,StdF,gamma):
    fitness = np.zeros(np.shape(label)[1])
    # print(np.shape(fitness))
    for i in range(Nc):
        TempSum = 0
        for j in range(Nc):
            if j != i:
                D = DisC[i,j]
                TempSum = TempSum + (math.exp(- (D**2) / StdF))**gamma
        fitness[i] = TempSum
    return fitness

def Pseduo_Peaks(DisC,Dist,label,fitness,StdF,gamma):

    # Search Stage of Pseduo Clusters at the temporal sample space
    NeiRad = 0.45*np.max(Dist)
    i = 0
    marked = []
    C_Indices = np.arange(1, np.shape(label)[1]+1) # The pseduo Cluster label of features
    PeakIndices = []
    Pfitness = []
    co = []
    F = fitness
    while True:

        PeakIndices.append(np.argmax(F))
        Pfitness.append(np.max(F))

        indices = NeighborSearch(DisC, label, PeakIndices[i], marked, NeiRad)

        C_Indices[indices] = PeakIndices[i]
        if len(indices) == 0:
            indices=[PeakIndices[i]]

        co.append(len(indices)) # Number of samples belong to the current
    # identified pseduo cluster
        marked = np.concatenate(([marked,indices]))

        # Fitness Proportionate Sharing
        tempF = Sharing(F, indices)

        F = tempF

        # Check whether all of samples has been assigned a pseduo cluster label
        if np.sum(co) >= (len(F)):

            break

        i=i+1 # Expand the size of the pseduo cluster set by 1
    return PeakIndices,Pfitness,C_Indices

def NeighborSearch(DisC, label, P_indice, marked, radius):
    Cluster = []
    for i in range(np.shape(label)[1]):
        if i not in marked:
            Dist = DisC[i, P_indice]
            if Dist <= radius:
                Cluster.append(i)
    Indices = Cluster

    return Indices

def Sharing(fitness, indices):
    newfitness = fitness
    sum1 = 0
    for j in range(len(indices)):
        sum1 = sum1 + fitness[indices[j]]
    for th in range(len(indices)):
            newfitness[indices[th]] = fitness[indices[th]] / (1+sum1)

    return newfitness

def Pseduo_Evolve(DisC, PeakIndices, PseDuoF, C_Indices, data, fitness, StdF, gamma):

    # Initialize the indices of Historical Pseduo Clusters and their fitness values
    HistCluster = PeakIndices
    HistClusterF = PseDuoF
    while True:
        # Call the merge function in each iteration
        [Cluster,Cfitness,F_Indices] = Pseduo_Merge(DisC, HistCluster, HistClusterF, C_Indices, data, fitness, StdF, gamma)
        # Check for the stablization of clutser evolution and exit the loop
        if len(np.unique(Cluster)) == len(np.unique(HistCluster)):
            break

        # Update the feature indices of historical pseduo feature clusters and
        # their corresponding fitness values

        HistCluster=Cluster
        HistClusterF=Cfitness
        C_Indices = F_Indices
    # Compute final evolved feature cluster information
    FCluster = Cluster
    Ffitness = Cfitness
    C_Indices = F_Indices

    return FCluster, Ffitness, C_Indices
#----------------------------------------------------------------------------------------------------------
def Pseduo_Merge(DisC, PeakIndices, PseDuoF, C_Indices, data, fitness, StdF, gamma):
    # Initialize the pseduo feature clusters lables for all features
    F_Indices = C_Indices
    ML = [] # Initialize the merge list as empty
    marked = [] #List of checked Pseduo Clusters Indices
    Unmarked = [] # List of unmerged Pseduo Clusters Indices
    for i in range(len(PeakIndices)):
            M = 1 # Set the merge flag as default zero
            MinDist = math.inf # Set the default Minimum distance between two feature clusters as infinite
            MinIndice = -1 # Set the default Neighboring feature cluster indices as zero
            # Check the current Pseduo Feature Cluster has been evaluated or not
            if PeakIndices[i] not in marked:
                for j in range(len(PeakIndices)):
                        if j != i:
                            # Divergence Calculation between two pseduo feature clusters
                            D = DisC[PeakIndices[i], PeakIndices[j]]
                            if MinDist > D:
                                MinDist = D
                                MinIndice = j
                if MinIndice>=0:
                    # Current feature pseduo cluster under check
                    Current = PeakIndices[i]
                    CurrentFit = PseDuoF[i]
                    # Neighboring feature pseduo cluster of the current checked cluster
                    Neighbor = PeakIndices[MinIndice]
                    NeighborFit = PseDuoF[MinIndice]

                    # A function to identify the bounady feature instance between two
                    # neighboring pseduo feature clusters
                    BP=Boundary_Points(DisC, F_Indices,data, PeakIndices[i], PeakIndices[MinIndice])
                    BPF=fitness[BP]
                    if BPF<1*min(CurrentFit,NeighborFit):
                        M=0 # Change the Merge flag

                    if M == 1:
                        ML.append([PeakIndices[i],PeakIndices[MinIndice]])
                        marked.append(PeakIndices[i])
                        marked.append(PeakIndices[MinIndice])
                    else:
                        Unmarked.append(PeakIndices[i])
                else:
                    Unmarked.append(PeakIndices[i])
    NewPI = []
    # Update the pseduo feature clusters list with the obtained mergelist
    for m in range(np.shape(ML)[0]):
        # print(ML[m][0],ML[m][1])
        if fitness[ML[m][0]] > fitness[ML[m][1]]:
            NewPI.append(ML[m][0])
            F_Indices[C_Indices==ML[m][1]] = ML[m][0]
        else:
            NewPI.append(ML[m][1])
            F_Indices[C_Indices==ML[m][0]] = ML[m][1]
    # Update the pseduo feature clusters list with pseduo clusters that have not appeared in the merge list
    for n in range(len(PeakIndices)):
        if PeakIndices[n] in Unmarked:
            NewPI.append(PeakIndices[n])

    # Updated pseduo feature clusters information after merging
    FCluster = np.unique(NewPI)
    Ffitness = fitness[FCluster]
    return FCluster, Ffitness, F_Indices

def Boundary_Points(DisC, F_Indices, data, Current, Neighbor):

    [N, dim] = np.shape(data)
    TempCluster1 = np.where(F_Indices == Current)
    TempCluster2 = np.where(F_Indices == Neighbor)

    TempCluster = np.append(TempCluster1,TempCluster2)

    D = []

    for i in range(len(TempCluster)):
        D1 = DisC[TempCluster[i], Current]
        D2 = DisC[TempCluster[i], Neighbor]

        D.append(abs(D1 - D2))

    FI = np.argmin(D)
    BD = TempCluster[FI]

    return BD


#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    label = Input()
    [N,Nc] = np.shape(label)

    [DisC,Dist] =  Feature_Dist(label,Nc)
    StdF = max(Dist)
    gamma = 5

    fitness = fitness_cal(DisC,Nc,label,StdF,gamma)
    oldfitness = np.copy(fitness)

    [PeakIndices,Pfitness,C_Indices] = Pseduo_Peaks(DisC,Dist,label,fitness,StdF,gamma)
    fitness = oldfitness

    # Pseduo Clusters Infomormation Extraction
    PseDuo = PeakIndices # Pseduo Feature Cluster centers
    PseDuoF = Pfitness # Pseduo Feature Clusters fitness values
    #-------------Check for possible merges among pseduo clusters-----------#

    [FCluster,Ffitness,C_Indices] = Pseduo_Evolve(DisC, PseDuo, PseDuoF, C_Indices, label, fitness, StdF, gamma)

    SF = FCluster

    end = time.time()
    print('The total time in seconds:', start-end)
