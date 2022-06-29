def  partition( arr,  low,  high) :
    pivot = arr[high]
    i = (low - 1)
    j = low
    while (j < high) :
        if (arr[j] <= pivot) :
            i += 1
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
        j += 1
    temp = arr[i + 1]
    arr[i + 1] = arr[high]
    arr[high] = temp


def sort( arr,  low,  high) :
    if (low < high) :
        pi = partition(arr, low, high)
        sort(arr, low, pi - 1)
        sort(arr, pi + 1, high)