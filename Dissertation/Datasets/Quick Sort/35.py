def quickSort( arr,  left,  right) :
    if (left < right) :
        index = partition(arr, left, right)
        quickSort(arr, left, index - 1)
        quickSort(arr, index + 1, right)

def  partition( arr,  left,  right) :
    pivot = arr[int((left + right) / 2)]
    while (left <= right) :
        while (arr[left] < pivot) :
            left += 1
        while (arr[right] > pivot) :
            right -= 1
        if (left <= right) :
            temp = arr[left]
            arr[left] = arr[right]
            arr[right] = temp
            left += 1
            right -= 1
    return left