def quickSort( array) :
    quickSort(array, 0, len(array) - 1)


def quickSort( array,  start,  end) :
    if (start >= end) :
        return
    partProv = partition(array, start, end)
    quickSort(array, start, partProv - 1)
    quickSort(array, partProv + 1, end)


def  partition( array,  start,  end) :
    resultIndex = start - 1
    compareKey = array[end]
    i = start
    while (i < end - 1) :
        if (array[i] <= compareKey) :
            resultIndex += 1
            exchange(array, resultIndex, i)
        i += 1
    exchange(array, resultIndex + 1, end)
    return resultIndex + 1


def exchange( array,  left,  right) :
    temp = array[left]
    array[left] = array[right]
    array[right] = temp