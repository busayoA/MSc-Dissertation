def merge (arrayList, left, right):
    i = j = k = 0
    #Evaluates the nodes to sort
    while i < len (left) and j < len (right): 
        if left[i] < right[j]: 
            arrayList[k] = left[i] 
            i+=1
        else: 
            arrayList[k] = right[j] 
            j+=1
        k+=1
          
    while i < len (left): 
        arrayList[k] = left[i] 
        i+=1
        k+=1
        
    while j < len (right): 
        arrayList[k] = right[j] 
        j+=1
        k+=1

def mergeSort (arrayList):
    if len( arrayList) > 1:
        #Find the mid point
        middle = len (arrayList) // 2

        #Splits the list into two new lists
        left = arrayList[:middle]
        right = arrayList[middle:]

        #Recursively calls the function to continue splitting the list
        mergeSort (left)
        mergeSort (right)

        #Sorts and reconstructs the list
        merge (arrayList, left, right)