def  quicksort( intArray) :
    return quicksort(intArray, 0, len(intArray) - 1)


def  quicksort( arr,  low,  high) :
    if (low == high) :
        return arr
    pivotIndex = high
    pivot = arr[high]
    highIndex = high - 1
    lowIndex = low
    pivotChange = False
    while (pivotChange == False) :
        lVal = arr[lowIndex]
        hVal = arr[highIndex]
        if (lVal > pivot) :
            if (hVal < pivot) :
                arr = swap(arr, lowIndex, highIndex)
                lowIndex += 1
                highIndex -= 1
            else :
                x = highIndex
                while (x >= lowIndex) :
                    if (x == lowIndex) :
                        swap(arr, x, pivotIndex)
                        pivotChange = True
                        quicksort(arr, low, x - 1)
                        quicksort(arr, x + 1, pivotIndex)
                        break
                    if (arr[x] < pivot) :
                        swap(arr, x, lowIndex)
                        break
                    x -= 1
        else :
            x = lowIndex
            while (x <= highIndex) :
                if (x == highIndex) :
                    swap(arr, x, pivotIndex)
                    pivotChange = True
                    break
                if (arr[x] > pivot) :
                    lowIndex = x
                    lVal = arr[x]
                    break
                x += 1
    return arr

def  swap( array,  posX,  posY) :
    temp = array[posX]
    array[posX] = array[posY]
    array[posY] = temp
    return array