def sort(myList):
    for i in range(len(myList)):
        for j in range(1, len(myList)-i):
            if myList[j-1] > myList[j]:   
                temp = myList[j-1]
                myList[j-1] = myList[j]
                myList[j] = temp
    print(myList)

# myList = [9, 2, 4, 1, 4]
# sort(myList) # TESTING