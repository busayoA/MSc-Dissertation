def sort(myList):
    size = len(myList)
    for i in range(size-1):
        for j in range(size-1, i, -1):
            if myList[j] < myList[j-1]:
                temp = myList[j]
                myList[j] = myList[j-1]
                myList[j-1] = temp
    print(myList)

myList = [9, 2, 4, 1, 4]
sort(myList) # TESTING