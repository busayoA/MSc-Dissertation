def merge( arr,  l,  m,  r) :
    n1 = m - l + 1
    n2 = r - m
    L = [0] * (n1)
    R = [0] * (n2)
    i = 0
    while (i < n1):
        L[i] = arr[l + i]
        i += 1
    j = 0
    while (j < n2):
        R[j] = arr[m + 1 + j]
        j += 1
    i = 0
    j = 0
    k = l
    while (i < n1 and j < n2):
        if (L[i] <= R[j]):
            arr[k] = L[i]
            i += 1
        else :
            arr[k] = R[j]
            j += 1
        k += 1
    while (i < n1):
        arr[k] = L[i]
        i += 1
        k += 1
    while (j < n2):
        arr[k] = R[j]
        j += 1
        k += 1


def sort( arr,  l,  r) :
    if (l < r) :
        m = int((l + r) / 2)
        sort(arr, l, m)
        sort(arr, m + 1, r)
        merge(arr, l, m, r)