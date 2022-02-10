def swaps(thisList):
    makeSwap = False
    length = len(thisList)
    for i in range(length):
        for j in range(length-1):
            makeSwap = False
            if (thisList[j] > thisList[j+ 1]):
                doSwap(thisList, j, j + 1)
                makeSwap = True

        if makeSwap is True:
            return

def doSwap(thisList, i, j):
    tempSwap = thisList[i]
    thisList[i] = thisList[j]
    thisList[j] = temp

