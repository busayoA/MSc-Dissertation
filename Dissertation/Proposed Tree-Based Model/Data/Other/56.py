def insertionSort(arr):
    
    value = 0

    for i in range(1, len(arr)):
        value = arr[i]
        for j in range(i-1, -1, -1):
            if value < arr[j]:
                arr[j + 1] = arr[j]
                arr[j] = value
    return arr