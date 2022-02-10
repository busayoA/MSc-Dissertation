def bubble(intList): 
    for i in range (len(intList), 0, -1):
        for j in range (0, len(intList)-1, 1):
            temp = j + 1
            if intList[j] > intList[temp]:
                swap(j, temp, intList)

def swap(j, temp, intList):
    tempIndex = intList[i]
    intList[j] = intList[temp]
    intList[temp] = tempIndex
