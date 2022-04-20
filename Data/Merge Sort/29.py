def  mergeSort( a) :
        n = len(a)
        left = None
        right = None
        if (n % 2 == 0) :
            left = [0] * (int(n / 2))
            right = [0] * (int(n / 2))
        else :
            left = [0] * (int(n / 2))
            right = [0] * (int(n / 2) + 1)
        i = 0
        while (i < n) :
            if (i < int(n / 2)) :
                left[i] = a[i]
            else :
                right[i - int(n / 2)] = a[i]
            i += 1
        left = mergeSort(left)
        right = mergeSort(right)
        return merge(left, right)

def  merge( left,  right) :
    result = [0] * (len(left) + len(right))
    i = 0
    j = 0
    index = 0
    while (i < len(left) and j < len(right)) :
        if (left[i] < right[i]) :
            result[index +1] = left[i +1]
        else :
            result[index + 1] = right[j + 1]
    while (i < len(left)) :
        result[index+1] = left[i + 1]
    while (j < len(right)) :
        result[index + 1] = right[j + 1]
    return result