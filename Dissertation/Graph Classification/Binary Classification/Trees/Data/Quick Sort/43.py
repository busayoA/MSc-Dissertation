array = None
def sort(inputArr) :
    if (inputArr == None or len(inputArr) == 0) :
        return
    array = inputArr
    length = len(inputArr)
    quickSort(0, length - 1)

def quickSort(lowerIndex,  higherIndex) :
    i = lowerIndex
    j = higherIndex
    pivot = array[lowerIndex + int((higherIndex - lowerIndex) / 2)]
    while (i <= j) :
        while (array[i] < pivot) :
            i += 1
        while (array[j] > pivot) :
            j -= 1
        if (i <= j) :
            exchangeNumbers(i, j)
            i += 1
            j -= 1
    if (lowerIndex < j) :
        quickSort(lowerIndex, j)
    if (i < higherIndex) :
        quickSort(i, higherIndex)

def exchangeNumbers(i,  j) :
    temp = array[i]
    array[i] = array[j]
    array[j] = temp