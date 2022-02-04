def sort(myList):
    swap = True
    while swap is True:
        swap = False
        for i in range(len(myList)-1):
            if myList[i] > myList[i+1]:
                swap = True
                temp = myList[i]
                myList[i] = myList[i+1]
                myList[i+1] = temp
