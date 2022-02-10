def selectionSort(array, arrayLength):
    for i in range(0, arrayLength):
        minimum, index = array[i], i
        for j in range(i + 1, arrayLength):
            if minimum > array[j]:
                index = j
        numToSwap = array[i]
        array[i] = minimum
        array[index] = numToSwap

    return array