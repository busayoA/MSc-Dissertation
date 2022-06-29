def mergesort(self, arr) :
    size = len(arr)
    if (size < 2) :
        return
    mid = int(size / 2)
    left = [0] * (mid)
    right = [0] * (size - mid)
    i = 0
    while (i < mid) :
        left[i] = arr[i]
        i += 1
    i = mid
    while (i < size) :
        right[i - mid] = arr[i]
        i += 1
    mergesort(left)
    mergesort(right)
    merge(left, right, arr)

def merge(self, left,  right,  arr) :
    i = 0
    j = 0
    k = 0
    ls = len(left)
    rs = len(right)
    ns = len(arr)
    while (i < ls and j < rs) :
        if (left[i] < right[j]) :
            arr[k] = left[i]
            i += 1
        else :
            arr[k] = right[j]
            j += 1
        k += 1
    while (i < ls) :
        arr[k] = left[i]
        i += 1
        k += 1
    while (j < rs) :
        arr[k] = right[j]
        j += 1
        k += 1