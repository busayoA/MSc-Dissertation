def quicksort( array,  l,  r) :
    i = 0
    j = 0
    pivot = 0
    tmp = 0
    if (l < r) :
        i = l
        j = r
        pivot = array[int((l + r) / 2)]
        while (i <= j) :
            while (array[i] < pivot) :
                i += 1
            while (array[j] > pivot) :
                j -= 1
            if (i <= j) :
                tmp = array[i]
                array[i] = array[j]
                array[j] = tmp
                i += 1
                j -= 1
        quicksort(array, l, j)
        quicksort(array, i, r)