def sortList(entList):
    minimum, minimumIndex = 0, 0
    for i in range(0, len(entList)):
        minimum = entList[i]
        minimumIndex = i

        for j in range(i+1, len(entList)):
            if minimum > entList[j]:
                minimum = entList[j]
                minimumIndex = j
            
            if minimumIndex is not i:
                entList[minimumIndex] = entList[i]
                entList[i] = minimum