def merge( a,  startIndex,  midIndex,  endIndex) :
    leftArraySize = midIndex - startIndex + 1
    rightArraySize = endIndex - midIndex
    L = [0] * (leftArraySize + 1)
    R = [0] * (rightArraySize + 1)
    i = 0
    j = 1
    i = 0
    while (i < leftArraySize) :
        L[i] = a[startIndex + i]
        i += 1
    j = 0
    while (j < rightArraySize) :
        R[j] = a[midIndex + j + 1]
        j += 1
    L[leftArraySize] = []
    R[rightArraySize] = []
    i = 0
    j = 0
    k = startIndex
    while (k <= endIndex) :
        if (L[i] <= R[j]) :
            a[k] = L[i]
            i = i + 1
        else :
            a[k] = R[j]
            j = j + 1
        k += 1

def mergeSort( a,  p,  r) :
    q = 0
    if (p < r) :
        q = int((p + r) / 2)
        mergeSort(a, p, q)
        mergeSort(a, q + 1, r)
        merge(a, p, q, r)