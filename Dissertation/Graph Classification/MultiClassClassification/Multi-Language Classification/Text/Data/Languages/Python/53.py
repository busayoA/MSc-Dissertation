def mergeSort(arr,  l,  r) :
    if (l < r) :
        mid = int((l + r) / 2)
        self.mergeSort(arr, l, mid)
        self.mergeSort(arr, mid + 1, r)
        self.merge(arr, l, r, mid)

def merge(arr,  l,  r,  mid) :
    n1 = mid - l + 1
    n2 = r - mid
    L = [0] * (n1)
    R = [0] * (n2)
    i = 0
    while (i < n1) :
        L[i] = arr[l + i]
        i += 1
    j = 0
    while (j < n2) :
        R[j] = arr[mid + j + 1]
        j += 1
    i = 0
    j = 0
    k = l
    while (i < n1 and j < n2) :
        if (L[i] <= R[j]) :
            arr[k] = L[i]
            i += 1
        else :
            arr[k] = R[j]
            j += 1
        k += 1
    while (i < n1) :
        arr[k] = L[i]
        i += 1
        k += 1
    while (j < n2) :
        arr[k] = R[j]
        j += 1
        k += 1