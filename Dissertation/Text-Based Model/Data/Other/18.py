def sorting(myList):
    min, ind = 0, 0
    for i in range(0, len(myList)):
        min = myList[i]
        ind = i

        for j in range(i+1, len(myList)):
            if min > myList[j]:
                min = myList[j]
                ind = j
            
            if ind is not i:
                myList[ind] = myList[i]
                myList[i] = min