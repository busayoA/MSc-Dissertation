def bubbleSort(array):
    length=len(array)
    result = True
    global count
    while result:	
        result = False
        i=0
        while (i < length-1):
            if (array[i] > array[i+1]):
                tempVar = array[i]
                array[i] = array[i+1]
                array[i+1] = tempVar
                result = True
            i=i+1
            count+=1
            print ("Sorting: " + str(array))
    return array