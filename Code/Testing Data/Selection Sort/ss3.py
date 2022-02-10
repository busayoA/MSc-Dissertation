def makeSelectionSort(arrList):
    for number in range(0, len(arrList)-1, 1):
        indexTemp = number
        for nextNumber in range(number+1, len(arrList), 1):
            if arrList[j] < arrList[index]:
                indexTemp = nextNumber
    
        lowerNumber = arrList[indexTemp]; 
        arrList[indexTemp] = arrList[number]
        arrList[number] = smallerNumber
    return arrList
