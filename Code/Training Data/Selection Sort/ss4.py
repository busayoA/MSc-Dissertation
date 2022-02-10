def sort(listList):
    temp, minimumIndex = 0, 0
    for i in range(len(listList)-1):
        minimumIndex = i
        for j in range(i+1, len(listList)):

            if listList[minimumIndex] > listList[j]:
                minimumIndex = j

        temp = listList[minimumIndex]
        listList[minimumIndex] = listList[i]
        listList[i] = temp
  