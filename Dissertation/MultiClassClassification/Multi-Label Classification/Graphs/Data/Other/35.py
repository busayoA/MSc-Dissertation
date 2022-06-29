def sort(thisList):
    listLength = len(thisList)
    for num in range(0, listLength, 1):
        temp = num
        for newNum in range(num+1, listLength, 1):
            if thisList[newNum] < thisList[temp]:
                temp = newNum
 
        thisList[num] = thisList[temp]
        thisList[temp] = thisList[num]