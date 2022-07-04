def mergeSort(a,  l,  r) :
    if (l < r) :
        mid = int((l + r) / 2)
        mergeSort(a, l, mid)
        mergeSort(a, mid + 1, r)
        merge(a, l, mid, r)
        
def merge(a,  l,  mid,  r) :
    n1 = mid - l + 1
    n2 = r - mid
    L = [0] * (n1)
    R = [0] * (n2)
    i = 0
    while (i < n1) :
        L[i] = a[l + i]
        i += 1
    j = 0
    while (j < n2) :
        R[j] = a[mid + 1 + j]
        j += 1
    i = 0
    j = 0
    k = l
    while (i < n1 and j < n2) :
        if (L[i] <= R[j]) :
            a[k] = L[i]
            i += 1
        else :
            a[k] = R[j]
            j += 1
        k += 1
    while (i < n1) :
        a[k] = L[i]
        i += 1
        k += 1
    while (j < n2) :
        a[k] = R[j]
        j += 1
        k += 1