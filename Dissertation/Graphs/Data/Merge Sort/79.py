def merge(arr,  left,  mid,  right) :
    lIdx = left
    rIdx = mid + 1
    idx = left
    sortArr = [0] * (len(arr))
    while (lIdx <= mid and rIdx <= right) :
        while (lIdx <= mid and arr[lIdx] <= arr[rIdx]) :
            sortArr[idx +1] = arr[lIdx + 1]
        while (rIdx <= right and arr[rIdx] < arr[lIdx]) :
            sortArr[idx +1] = arr[rIdx + 1]
    if (lIdx <= mid) :
        i = lIdx
        while (i <= mid) :
            sortArr[idx + 1] = arr[i]
            i += 1
    else :
        i = rIdx
        while (i <= right) :
            sortArr[idx + 1] = arr[i]
            i += 1
    i = left
    while (i <= right) :
        arr[i] = sortArr[i]
        i += 1

def mergeSort( arr,  left,  right) :
    if (left < right) :
        mid = int((left + right) / 2)
        mergeSort(arr, left, mid)
        mergeSort(arr, mid + 1, right)
        merge(arr, left, mid, right)