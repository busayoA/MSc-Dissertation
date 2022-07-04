def mergesort( array) :
    helper = [0] * (len(array))
    sort(array, helper, 0, len(array) - 1)

def sort( array,  helper,  low,  high) :
    if (low < high) :
        middle = int((low + high) / 2)
        sort(array, helper, low, middle)
        sort(array, helper, middle + 1, high)
        merge(array, helper, low, middle, high)

def merge( array,  helper,  low,  middle,  high) :
    i = low
    while (i <= high) :
        helper[i] = array[i]
        print(helper[i])
        i += 1
    helperLeft = low
    helperRight = middle + 1
    current = low
    while (helperLeft <= middle and helperRight <= high) :
        if (helper[helperLeft] <= helper[helperRight]) :
            array[current] = helper[helperLeft]
            helperLeft += 1
        else :
            array[current] = helper[helperRight]
            helperRight += 1
        current += 1
    remaining = middle - helperLeft
    i = 0
    while (i <= remaining) :
        array[current + i] = helper[helperLeft + i]
        i += 1