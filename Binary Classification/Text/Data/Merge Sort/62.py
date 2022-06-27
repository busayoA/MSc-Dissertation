def mergesort( array) :
    mergesort(array, [0] * (len(array)), 0, len(array) - 1)

def mergesort( array,  temp,  start,  end) :
    if (start >= end) :
        return
    if (end - start == 1 and array[start] > array[end]) :
        swap(array, start, end)
        return
    middle = int((start + end) / 2)
    mergesort(array, temp, start, middle)
    mergesort(array, temp, middle + 1, end)
    merge(array, temp, start, middle, middle + 1, end)


def merge( array,  temp,  leftStart,  leftEnd,  rightStart,  rightEnd) :
    size = (leftEnd - leftStart) + (rightEnd - rightStart) + 2
    i = leftStart
    j = rightStart
    m = 0
    while (m < size) :
        if (i > leftEnd) :
            temp[m] = array[j]
            j += 1
        elif(j > rightEnd) :
            temp[m] = array[i]
            i += 1
        elif(array[i] < array[j]) :
            temp[m] = array[i]
            i += 1
        else :
            temp[m] = array[j]
            j += 1
        m += 1
    m = 0
    while (m < size) :
        array[m + leftStart] = temp[m]
        m += 1


def swap( array,  i,  j) :
    temp = array[i]
    array[i] = array[j]
    array[j] = temp