def selection(intList):
    length = len(intList)
    for index in range(0, length-1, 1):
        for nextIndex in range(index+1, length, 1):
            if intList[index] > intList[nextIndex]:

                doSelectionSwap(intList, index, nextIndex)
    print(intList)


def doSelectionSwap(intList, index, nextIndex):
    temp = intList[index]
    intList[index] = intList[nextIndex]
    intList[nextIndex] = temp