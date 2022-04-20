def swapArrayElements(array,  indexa,  indexb) :
    tmp = array[indexa]
    array[indexa] = array[indexb]
    array[indexb] = tmp

def  partition( array,  low,  high) :
    pivot = array[high]
    result = 0
    left = low
    right = high - 1
    while (left < right) :
        while (left < right and array[left] < pivot) :
            left += 1
        while (left < right and array[right] >= pivot) :
            right -= 1
        if (left != right) :
            swapArrayElements(array, left, right)
    if (array[left] >= pivot) :
        swapArrayElements(array, left, high)
        result = left
    else :
        result = high
    return result

def quickSort(arrayToSort,  low,  high) :
    if (low >= high) :
        return
    pivotPosition = partition(arrayToSort, low, high)
    quickSort(arrayToSort, low, pivotPosition - 1)
    quickSort(arrayToSort, pivotPosition + 1, high)