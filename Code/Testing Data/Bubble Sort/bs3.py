def sortList(bubbleList):
    temp, length = 0, len(bubbleList)
    for i in range(length, 0, -1):
        for j in range(0, length-1, 1):
            temp = j + 1
            if bubbleList[j] > bubbleList[temp]:
                makeSwap(j, temp, bubbleList)
        print(bubbleList)

def makeSwap(j, temp, bubbleList):
    tempNum = bubbleList[j]
    bubbleList[j] = bubbleList[temp]
    bubbleList[temp] = tempNum
