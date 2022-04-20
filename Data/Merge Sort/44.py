def mergeSort( A) :
    mergeSort(A, 0, len(A) - 1)

def mergeSort( A,  l,  r) :
    if (l < r) :
        m = l + int((r - l) / 2)
        mergeSort(A, l, m)
        mergeSort(A, m + 1, r)
        merge(A, l, m, r)

def merge( A,  l,  m,  r) :
    n1 = m - l + 1
    n2 = r - m
    left = [0] * (n1)
    right = [0] * (n2)
    i = 0
    while (i < n1) :
        left[i] = A[l + i]
        i += 1
    i = 0
    while (i < n2) :
        right[i] = A[m + 1 + i]
        i += 1
    i = 0
    j = 0
    k = l
    while (i < n1 and j < n2) :
        if (left[i] <= right[j]) :
            A[k] = left[i]
            i += 1
        else :
            A[k] = right[j]
            j += 1
        k += 1
    if (i == n1) :
        while (j < n2) :
            A[k] = right[j]
            j += 1
            k += 1
    if (j == n2) :
        while (i < n1) :
            A[k] = left[i]
            i += 1
            k += 1