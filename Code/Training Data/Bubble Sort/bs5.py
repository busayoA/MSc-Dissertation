def sort(myList):
    size = len(myList)
    for i in range(size-1):
        for j in range(size-1, i, -1):
            if myList[j] < myList[j-1]:
                temp = myList[j]
                myList[j] = myList[j-1]
                myList[j-1] = temp
