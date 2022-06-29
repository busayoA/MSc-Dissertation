def quicksort( arr,  l,  h) :
    if (l >= h) :
        return
    v = partition(arr, l, h)
    quicksort(arr, l, v - 1)
    quicksort(arr, v + 1, h)

def  partition( arr,  l,  h) :
    lastSwap = l + 1
    pivot = arr[l]
    i = l + 1
    while (i <= h) :
        if (arr[i] < pivot) :
            swap(arr, lastSwap + 1, i)
        i += 1
    swap(arr, l, lastSwap - 1)
    return lastSwap - 1

def swap( arr,  i,  j) :
    t = arr[i]
    arr[i] = arr[j]
    arr[j] = t