def quickSort2( A) :
    quickSort(A, 0, len(A) - 1)


def quickSort( A,  l,  r) :
    if (l < r) :
        pivot = partition(A, l, r)
        quickSort(A, l, pivot - 1)
        quickSort(A, pivot + 1, r)


def  partition( A,  l,  r) :
    pivot = A[r]
    i = l
    j = l
    while (j < r) :
        if (A[j] <= pivot) :
            t = A[j]
            A[j] = A[i]
            A[i] = t
            i += 1
        j += 1
    A[r] = A[i]
    A[i] = pivot
    return i