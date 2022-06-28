def sort(myList): 
     for i in range (len(myList), 0, -1):
         for j in range (0, len(myList)-1, 1):
             temp = j + 1
             if myList[j] > myList[temp]:
                 swap(j, temp, myList)

def swap(j, temp, myList):
     tempIndex = myList[i]
     myList[j] = myList[temp]
     myList[temp] = tempIndex