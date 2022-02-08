def sort(myList):
    for i in range(len(myList)):
        for j in range(1, len(myList)-i):
            if myList[j-1] > myList[j]:   
                temp = myList[j-1]
                myList[j-1] = myList[j]
                myList[j] = temp