def sort(myList):
    minimum, temp = 0, 0

    for i in range(0, len(myList)-1):
        minimum = i
        for j in range(i+1, len(myList)):
            if myList[j] < myList[minimum]:
                minimum = j

        temp = myList[i]
        myList[i] = myList[minimum]
        myList[minimum] = temp