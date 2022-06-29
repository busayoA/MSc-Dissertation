def  partition(arr,  left,  right) :
    i = left
    j = right
    tmp = 0
    pivot = arr[int((left + right) / 2)]
    while (i <= j) :
        while (arr[i] < pivot) :
            i += 1
        while (arr[j] > pivot) :
            j -= 1
        if (i <= j) :
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
            i += 1
            j -= 1
    return i

def  quickSort(arr,  left,  right) :
    index = partition(arr, left, right)
    if (left < index - 1) :
        quickSort(arr, left, index - 1)
    if (index < right) :
        quickSort(arr, index, right)
    return arr