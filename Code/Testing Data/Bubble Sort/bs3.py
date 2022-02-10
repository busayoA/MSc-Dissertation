def sort(myList):
    temp, length = 0, len(myList)
    for i in range(length, 0, -1):
        for j in range(0, length-1, 1):
            temp = j + 1
            if myList[j] > myList[temp]:
                makeSwap(j, temp, myList)
        print(myList)

def makeSwap(j, temp, myList):
    tempNum = myList[j]
    myList[j] = myList[temp]
    myList[temp] = tempNum
