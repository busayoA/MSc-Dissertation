def merge(arr,  l,  m,  r) :
    len1 = m - l + 1
    len2 = r - m
    left = [0] * (len1)
    right = [0] * (len2)
    i = 0
    while (i < len1) :
        left[i] = arr[l + i]
        i += 1
    print("Left array: ", end ="")
    j = 0
    while (j < len2) :
        right[j] = arr[m + 1 + j]
        j += 1
    i = 0
    j = 0
    k = l
    while (i < len1 and j < len2) :
        if (left[i] <= right[j]) :
            arr[k] = left[i]
            i += 1
        else :
            arr[k] = right[j]
            j += 1
        k += 1
    while (i < len1) :
        arr[k] = left[i]
        i += 1
        k += 1
    while (j < len2) :
        arr[k] = right[j]
        j += 1
        k += 1

def sort(arr,  l,  r) :
    if (l < r) :
        m = int((l + r) / 2)
        sort(arr, l, m)
        sort(arr, m + 1, r)
        merge(arr, l, m, r)