tempMergArr = []
array = []
def sort(inputArr) :
    array = inputArr
    length = len(inputArr)
    tempMergArr = [0] * (length)
    doMergeSort(0, length - 1)

def doMergeSort(lowerIndex,  higherIndex) :
    if (lowerIndex < higherIndex) :
        middle = lowerIndex + int((higherIndex - lowerIndex) / 2)
        doMergeSort(lowerIndex, middle)
        doMergeSort(middle + 1, higherIndex)
        mergeParts(lowerIndex, middle, higherIndex)

def mergeParts(lowerIndex,  middle,  higherIndex) :
    i = lowerIndex
    while (i <= higherIndex) :
        tempMergArr.append(array[i])
        i += 1
    i = lowerIndex
    j = middle + 1
    k = lowerIndex
    while (i <= middle and j <= higherIndex) :
        if (tempMergArr[i] <= tempMergArr[j]) :
            array[k] = tempMergArr[i]
            i += 1
        else :
            array[k] = tempMergArr[j]
            j += 1
        k += 1
    while (i <= middle) :
        array[k] = tempMergArr[i]
        k += 1
        i += 1