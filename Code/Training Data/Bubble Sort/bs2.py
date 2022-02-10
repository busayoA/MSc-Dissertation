def bubbleSort(myList):
    swaps, temp, sortedEnd = 0, 0, len(myList)-1
    for i in range(len(myList)):
        for j in range(sortedEnd):
            if  myList[j] > myList[j + 1]:
                    temp = myList[j]
                    myList[j] = myList[j + 1]
                    myList[j + 1] = temp
                    swaps += 1
        sortedEnd -= 1
