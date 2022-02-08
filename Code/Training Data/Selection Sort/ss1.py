def sort(myList):
    minimum, minimumIndex = 0, 0
    for i in range(0, len(myList)):
        minimum = myList[i]
        minimumIndex = i

        for j in range(i+1, len(myList)):
            if minimum > myList[j]:
                minimum = myList[j]
                minimumIndex = j
            
            if minimumIndex is not i:
                myList[minimumIndex] = myList[i]
                myList[i] = minimum
