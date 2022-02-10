def sort(aList):
    for i in range(len(aList)):
        for j in range(len(aList)-1):
            if aList[j] > aList[j+1]:
                temporary = aList[j]
                aList[j] = aList[j+1]
                aList[j+1] = temporary
