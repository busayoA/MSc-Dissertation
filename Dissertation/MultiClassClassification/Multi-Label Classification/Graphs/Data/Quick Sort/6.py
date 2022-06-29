def quickSort(listToSort):
     if len(listToSort) == 1 or len(listToSort) == 0:
         return listToSort
     elif len(listToSort) == 2:
         if listToSort[0] > listToSort[1]:
             temp = listToSort[0]
             listToSort[0] = listToSort[1]
             listToSort[1] = temp
         return listToSort

     midNum = listToSort[0] # Seleting a middle number
     currentIndex = 0 # Record the current index of the middle number
     for i in range(1,len(listToSort)): # iterating through the rest of the array
         if listToSort[i] < midNum:
             temp = listToSort[i]
             for j in reversed(range(currentIndex,i)): # moving previous elements up a space for this number to go to the front
                 listToSort[j + 1] = listToSort[j]
             listToSort[currentIndex] = temp
             currentIndex += 1
     # Split the array into two and start recursion
     return quickSort(listToSort[0:currentIndex]) + [midNum] + quickSort(listToSort[currentIndex+1:])