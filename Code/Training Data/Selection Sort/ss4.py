def sort(myList):
    temp, minimumIndex = 0, 0
    for i in range(len(myList)-1):
        minimumIndex = i
        for j in range(i+1, len(myList)):

            if myList[minimumIndex] > myList[j]:
                minimumIndex = j

        temp = myList[minimumIndex]
        myList[minimumIndex] = myList[i]
        myList[i] = temp
  