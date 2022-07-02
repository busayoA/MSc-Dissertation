def quickSort(a,  low,  high) :
    if (low < high) :
        p = partition(a, low, high)
        quickSort(a, low, p - 1)
        quickSort(a, p + 1, high)

def swap(a,  i,  j) :
    temp = a[i]
    a[i] = a[j]
    a[j] = temp

def  partition(a,  low,  high) :
    pivot = a[high]
    i = low - 1
    j = low
    while (j <= high - 1) :
        if (a[j] <= pivot) :
            i += 1
            swap(a, i, j)
        j += 1
    swap(a, i + 1, high)
    return (i + 1)