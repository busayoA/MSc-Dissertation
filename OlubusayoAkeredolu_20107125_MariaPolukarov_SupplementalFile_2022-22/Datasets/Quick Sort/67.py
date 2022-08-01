def sort( array,  left,  right) :
    if (left >= right) :
        return
    pivotIndex = partition(array, left, right)
    sort(array, left, pivotIndex - 1)
    sort(array, pivotIndex + 1, right)


def swap( x,  y) :
    temp = x
    x = y
    y = temp


def  partition( array,  left,  right) :
    pivot = array[right]
    partitionIndex = left
    tempData = 0
    i = left
    while (i < right) :
        if (array[i] > pivot) :
            pass
        else :
            tempData = array[i]
            array[i] = array[partitionIndex]
            array[partitionIndex] = tempData
            partitionIndex += 1
        i += 1
    tempData = array[partitionIndex]
    array[partitionIndex] = array[right]
    array[right] = tempData
    return partitionIndex