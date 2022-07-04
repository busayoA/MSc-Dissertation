number, numbers = 0, []
def sort( values) :
    if (values == None or len(values) == 0) :
        return
    numbers = values
    number = len(values)
    quicksort(0, number - 1)

def quicksort(low,  high) :
    i = low
    j = high
    pivot = numbers[low + int((high - low) / 2)]
    while (i <= j) :
        while (numbers[i] < pivot) :
            i += 1
        while (numbers[j] > pivot) :
            j -= 1
        if (i <= j) :
            exchange(i, j)
            i += 1
            j -= 1
    if (low < j) :
        quicksort(low, j)
    if (i < high) :
        quicksort(i, high)

def exchange(i,  j) :
    temp = numbers[i]
    numbers[i] = numbers[j]
    numbers[j] = temp