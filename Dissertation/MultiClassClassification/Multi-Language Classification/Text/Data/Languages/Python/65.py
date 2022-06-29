def mergesort( low,  high) :
    if (low < high) :
        middle = low + int((high - low) / 2)
        mergesort(low, middle)
        mergesort(middle + 1, high)
        merge(low, middle, high, [0] * (10))
        
def merge(low,  middle,  high,  numbers) :
    helper = [0] * (len(numbers))
    i = low
    while (i <= high) :
        helper[i] = numbers[i]
        i += 1
    i = low
    j = middle + 1
    k = low
    while (i <= middle and j <= high) :
        if (helper[i] <= helper[j]) :
            numbers[k] = helper[i]
            i += 1
        else :
            numbers[k] = helper[j]
            j += 1
        k += 1
    while (i <= middle) :
        numbers[k] = helper[i]
        k += 1
        i += 1