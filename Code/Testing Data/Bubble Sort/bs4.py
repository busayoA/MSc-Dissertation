def sort(myList):
    makeSwap = False
    length = len(myList)
    for i in range(length):
        for j in range(length-1):
            makeSwap = False
            if (myList[j] > myList[j+ 1]):
                doSwap(myList, j, j + 1)
                makeSwap = True

        if makeSwap is True:
            return

def doSwap(myList, i, j):
    tempSwap = myList[i]
    myList[i] = myList[j]
    myList[j] = temp

