def sorting(myList, minimum, maximum):
    
    if minimum == len(myList-1):
        return
    if maximum== len(myList):
        sorting(myList, minimum+1, minimum+2)
        return
    if myList[minimum] > myList[maximum]:
        temp = myList[minimum]
        myList[minimum] = myList[maximum]
        myList[maximum] = temp

    sorting(myList, minimum, maximum+1)