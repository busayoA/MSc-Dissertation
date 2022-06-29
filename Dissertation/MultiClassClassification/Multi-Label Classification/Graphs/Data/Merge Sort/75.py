def merge( arr,  p,  q,  r) :
    lLen = q - p + 1
    rLen = r - q
    L = [0] * (lLen + 1)
    R = [0] * (rLen + 1)
    i = 0
    while (i < lLen) :
        L[i] = arr[p + i]
        i += 1
    j = 0
    while (j < rLen) :
        R[j] = arr[q + j + 1]
        j += 1
    L[lLen] = []
    R[rLen] = []
    i = 0
    j = 0
    k = p
    while (k <= r) :
        if (L[i] <= R[j]) :
            arr[k] = L[i]
            i += 1
        else :
            arr[k] = R[j]
            j += 1
        k += 1

def mergeSort( arr,  p,  r) :
    if (p < r) :
        q = int((p + r) / 2)
        mergeSort(arr, p, q)
        mergeSort(arr, q + 1, r)
        merge(arr, p, q, r)