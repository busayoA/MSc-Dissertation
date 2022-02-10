def sorting(myList, minimum, maximum):
    
    if minimum == len(myList-1):
        return
    if maximum== len(myList):
        sort(myList, minimum+1, minimum+2)
        return
    if myList[minimum] > myList[maximum]:
        temp = myList[minimum]
        myList[minimum] = myList[maximum]
        myList[maximum] = temp

    sort(myList, minimum, maximum+1)