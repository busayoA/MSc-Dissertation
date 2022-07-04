counter = 0
def quickSort(a) :
    quickSort(a, 0, len(a) - 1)

def quickSort(a,  begin,  end) :
    if (end - begin <= 1) :
        return
    counter += (end - begin)
    pivotIndex = partition(a, begin, end)
    quickSort(a, begin, pivotIndex - 1)
    quickSort(a, pivotIndex + 1, end)

def  partition(a,  l,  r) :
    p = a[l]
    i = l + 1
    j = l + 1
    while (j <= r) :
        if (a[j] < p) :
            swap(a, j, i)
            i += 1
        j += 1
    newPivotIndex = i - 1
    swap(a, l, newPivotIndex)
    return newPivotIndex

def swap(a,  i,  j) :
    buf = a[i]
    a[i] = a[j]
    a[j] = buf