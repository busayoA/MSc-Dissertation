def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
    # this is the number of loops we do
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
