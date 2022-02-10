def sort(myList):
    print(myList)
    temp = 0
    while len(myList) > 0:
        index = 0
        while index < (len(myList)-1):
            if myList[index] > myList[index + 1]:
                temp = myList[index]
                myList[index] = myList[index + 1]
                myList[index + 1] = temp
            index += 1
            print(myList)
