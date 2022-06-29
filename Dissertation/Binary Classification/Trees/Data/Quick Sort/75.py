def quickSort( arr,  low,  high) :
    if (arr == None or len(arr) == 0) :
        return
    if (low > high) :
        return
    middle = low + int((high - low) / 2)
    pivot = arr[middle]
    i = low
    j = high
    while (i <= j) :
        while (arr[i] < pivot) :
            i += 1
        while (arr[j] > pivot) :
            j -= 1
        if (i <= j) :
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            i += 1
            j -= 1
    if (low < j) :
        quickSort(arr, low, j)
    if (high > i) :
        quickSort(arr, i, high)