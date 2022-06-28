def sort(myList):
     for i in range(len(myList)):
         for j in range(len(myList)-1-i):
             if myList[j] > myList[j+1]:
                 makeSwap(myList, j, j + 1)
     print(myList)


def makeSwap(myList, i, j):
     temp = myList[i]
     myList[i] = myList[j]
     myList[j] = temp