def sort( arr) :
    sort(arr, 0, len(arr) - 1)

def sort( arr,  left,  right) :
    if (left >= right) :
        return
    mid = int((left + right) / 2)
    sort(arr, left, mid)
    sort(arr, mid + 1, right)
    merge(arr, left, right)

def merge( arr,  left,  right) :
    l = left
    mid = int((left + right) / 2)
    midRight = mid + 1
    index = 0
    tmp = [0] * (right - left + 1)
    while (left <= mid and midRight <= right) :
        if (arr[left] <= arr[midRight]) :
            tmp[index] = arr[left]
            left += 1
        else :
            tmp[index] = arr[midRight]
            midRight += 1
        index += 1
    i = 0
    while (i < len(tmp)) :
        arr[i + l] = tmp[i]
        i += 1