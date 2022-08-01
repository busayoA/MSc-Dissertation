def quickSort( arr,  x,  y) :
    if (x >= y) :
        return
    pivot = partition(arr, x, y)
    quickSort(arr, x, pivot)
    quickSort(arr, pivot + 1, y)


def  partition( arr,  x,  y) :
    pivot = arr[x]
    while (x < y) :
        while (arr[x] < pivot) :
            x += 1
        while (arr[y] > pivot) :
            y -= 1
        swapNumbers(x, y, arr)
    return x


def swapNumbers( x,  y,  arr) :
    temp = arr[x]
    arr[x] = arr[y]
    arr[y] = temp