length = 0
pivot = 0
array = None
def sort(inputArr) :
    if (inputArr == None or len(inputArr) == 0) :
        return
    array = inputArr
    length = len(inputArr)
    pivot = len(array) - 1
    quickSort(array, length - 1)

def quickSort(array,  higherIndex) :
    i = 0
    if (array[i] > pivot) :
        temp = array[i]
        a = i
        while (a < pivot) :
            array[a] = array[a + 1]
            pivot = a
            a += 1
        array[len(array) - 1] = temp
        i += 1
    if (0 < i) :
        newArray = [0] * (len(array) - 2)
        a = 1
        while (a < len(array)) :
            newArray[a - 1] = array[a]
            a += 1
        quickSort(newArray, higherIndex)
    else :
        newArray = [0] * (len(array) - 2)
        a = 1
        while (a < len(array)) :
            newArray[a - 1] = array[a]
            a += 1
        quickSort(array, higherIndex)