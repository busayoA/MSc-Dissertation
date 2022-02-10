def sortingAlgorithm(sortList):
    for i in range(len(sortList)):
        for j in range(1, len(sortList)-i):
            if sortList[j-1] > sortList[j]:   
                temp = sortList[j-1]
                sortList[j-1] = sortList[j]
                sortList[j] = temp