def sortBubble(bList):
    size = len(bList)
    for i in range(size-1):
        for j in range(size-1, i, -1):
            if bList[j] < bList[j-1]:
                temp = bList[j]
                bList[j] = bList[j-1]
                bList[j-1] = temp
