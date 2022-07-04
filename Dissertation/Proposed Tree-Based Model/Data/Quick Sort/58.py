def quickSort( numbers) :
    quickSort(numbers, 0, len(numbers) - 1)


def quickSort( a,  p,  r) :
    if (p < r) :
        q = partition(a, p, r)
        quickSort(a, p, q)
        quickSort(a, q + 1, r)


def  partition( a,  p,  r) :
    x = a[p]
    i = p - 1
    j = r + 1
    while (True) :
        i += 1
        while (i < r and a[i] < x) :
            i += 1
        j -= 1
        while (j > p and a[j] > x) :
            j -= 1
        if (i < j) :
            swap(a, i, j)
        else :
            return j


def swap( a,  i,  j) :
    temp = a[i]
    a[i] = a[j]
    a[j] = temp