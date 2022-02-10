def sort(arr):
    length = len(arr)
    for n in range(length):
        smallerNumber, smallerIndex = arr[n], n
        for k in range(n, length):
            if arr[k] < arr[smallerIndex]:
                smallerNumber = arr[k]
                smallerIndex = j
        
        swapItems(arr, n, smallerIndex)

def swapItems(arr, a, b):
    swap = arr[a]
    arr[a] = arr[b]
    arr[b] = swap
