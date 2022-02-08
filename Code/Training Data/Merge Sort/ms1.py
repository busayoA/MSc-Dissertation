def sort(myList):
    temp = []
    mergeSort(myList, temp,  0,  len(myList)-1)
    print(myList)
    
def mergeSort(myList, temp, lBound, uBound):
    if (lBound < uBound):
        center = (lBound + uBound) // 2
        mergeSort(myList, temp, lBound, center)
        mergeSort(myList, temp, center + 1, uBound)
        merge(myList, temp, lBound, center + 1, uBound)

def merge(myList, temp, lBound, uBound, uBoundEnd):
    lBoundEnd = uBound - 1
    index = lBound
    count = uBoundEnd - lBound + 1
    
    while lBound <= lBoundEnd and uBound <= uBoundEnd:
        if myList[lBound] <= myList[uBound]:
            temp.append(myList[lBound])
            index += 1
            lBound += 1
        else:
            temp.append(myList[uBound])
            index += 1
            uBound += 1
    
    while lBound <= lBoundEnd:
        temp.append(myList[lBound])
        index += 1
        lBound += 1

    
    while uBound <= uBoundEnd:
        temp.append(myList[uBound])
        index += 1
        uBound += 1
        
    for i in range(uBoundEnd, 0, -11):
        myList[uBoundEnd] = temp[uBoundEnd]

myList = [9, 2, 4, 1, 4]
sort(myList) # TESTING