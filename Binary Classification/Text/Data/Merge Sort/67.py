def sort( data) :
    length = len(data) - 1
    mergeSort(data, 0, length)


def mergeSort( data,  begin,  end) :
    if (begin < end) :
        mid = int((begin + end) / 2)
        mergeSort(data, begin, mid)
        mergeSort(data, mid + 1, end)
        merge(data, begin, mid, end)


def merge( data,  begin,  mid,  end) :
    leftIndex = begin
    rightIndex = mid + 1
    tmp = [None] * (end - begin + 1)
    tmpIndex = 0
    while (leftIndex <= mid and rightIndex <= end) :
        if (data[leftIndex] > data[rightIndex]) :
            tmp[tmpIndex] = data[rightIndex]
            rightIndex += 1
        else :
            tmp[tmpIndex] = data[leftIndex]
            leftIndex += 1
        tmpIndex += 1
    while (leftIndex <= mid) :
        tmp[tmpIndex] = data[leftIndex]
        tmpIndex += 1
        leftIndex += 1
    while (rightIndex <= end) :
        tmp[tmpIndex] = data[rightIndex]
        tmpIndex += 1
        rightIndex += 1
    i = 0
    while (i < tmpIndex) :
        data[begin + i] = tmp[i]
        i += 1