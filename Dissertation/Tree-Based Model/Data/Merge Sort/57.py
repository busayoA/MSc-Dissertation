def merge( arr,  p,  q,  r) :
    n1 = q - p + 1
    n2 = r - q
    left = [0] * (n1 + 1)
    right = [0] * (n2 + 1)
    i = 0
    while (i < n1) :
        left[i] = arr[p + i]
        i += 1
    i = 0
    while (i < n2) :
        right[i] = arr[q + i + 1]
        i += 1
    left[n1] = []
    right[n2] = []
    a = 0
    b = 0
    i = p
    while (i <= r) :
        if (left[a] <= right[b]) :
            arr[i] = left[a + 1]
        else :
            arr[i] = right[b + 1]
        i += 1

def mergeSort( arr,  p,  r) :
    if (p < r) :
        q = int((p + r) / 2)
        mergeSort(arr, p, q)
        mergeSort(arr, q + 1, r)
        merge(arr, p, q, r)