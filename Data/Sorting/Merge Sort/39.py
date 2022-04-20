def mergeSort( arr) :
        if (len(arr) > 1) :
            left = leftHalf(arr)
            right = rightHalf(arr)
            mergeSort(left)
            mergeSort(right)
            mergeArrays(arr, left, right)

def  leftHalf( arr) :
    size = int(len(arr) / 2)
    left = [0] * (size)
    i = 0
    while (i < size) :
        left[i] = arr[i]
        i += 1
    return left

def  rightHalf( arr) :
    size1 = int(len(arr) / 2)
    size2 = len(arr) - size1
    right = [0] * (size2)
    i = 0
    while (i < size2) :
        right[i] = arr[i + size1]
        i += 1
    return right

def mergeArrays( arr,  left,  right) :
    i1 = 0
    i2 = 0
    i = 0
    while (i < len(arr)) :
        if (i2 >= len(right) or (i1 < len(left) and left[i1] <= right[i2])) :
            arr[i] = left[i1]
            i1 += 1
        else :
            arr[i] = right[i2]
            i2 += 1
        i += 1