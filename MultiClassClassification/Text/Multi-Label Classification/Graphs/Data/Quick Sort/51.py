def  quickSort( array,  start,  end) :
    if (start >= end) :
        return 0
    pivotIndex = chooseMedianPivot(array, start, end)
    pivot = array.index(pivotIndex)
    swap(array, pivotIndex, start)
    wallIndex = start + 1
    i = start + 1
    while (i <= end) :
        if (array.index(i) < pivot) :
            swap(array, i, wallIndex)
            wallIndex += 1
        i += 1
    wallIndex -= 1
    swap(array, start, wallIndex)
    return end - start + quickSort(array, start, wallIndex - 1) + quickSort(array, wallIndex + 1, end)


def swap( array,  first,  second) :
    temp1 = array.index(first)
    temp2 = array.index(second)
    array.set(first,temp2)
    array.set(second,temp1)


def  chooseMedianPivot( array,  start,  end) :
    mid = int((start + end) / 2)
    min = min(array.index(mid),min(array.index(start),array.index(end)))
    max = max(array.index(mid),max(array.index(start),array.index(end)))
    median = array.index(mid) + array.index(start) + array.index(end) - min - max
    indexes = [start, mid, end]
    for i in indexes :
        if (array.index(i) == median) :
            return i
    return mid