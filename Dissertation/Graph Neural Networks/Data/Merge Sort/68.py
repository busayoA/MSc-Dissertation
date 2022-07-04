def  merge_sort( arr) :
    if (len(arr) <= 1) :
        return arr
    middle = int(len(arr) / 2)
    leftArray = [0] * (middle)
    rightArray = [0] * (len(arr) - middle)
    i = 0
    while (i < middle) :
        leftArray[i] = arr[i]
        i += 1
    rIndex = 0
    j = middle
    while (j < len(arr)) :
        rightArray[rIndex + 1] = arr[j]
        j += 1
    leftArray = merge_sort(leftArray)
    rightArray = merge_sort(rightArray)
    return merge(leftArray, rightArray)


def  merge( left,  right) :
    result = [0] * (len(left) + len(right))
    lIndex = 0
    rIndex = 0
    resultIndex = 0
    while (lIndex < len(left) or rIndex < len(right)) :
        if (lIndex < len(left) and rIndex < len(right)) :
            if (left[lIndex] <= right[rIndex]) :
                result[resultIndex +1] = left[lIndex + 1]
            else :
                result[resultIndex + 1] = right[rIndex + 1]
        else :
            if (lIndex < len(left)) :
                result[resultIndex + 1] = left[lIndex + 1]
            if (rIndex < len(right)) :
                result[resultIndex + 1] = right[rIndex + 1]
    return result