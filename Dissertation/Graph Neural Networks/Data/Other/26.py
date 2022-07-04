def shortSort(myList):
    for i in range(len(myList)):
        index = i
        for j in range(i+1, len(myList)):
            if myList[index] > myList[j]:
                index = j  
                
        myList[i] = myList[index]
        myList[index] = myList[i]