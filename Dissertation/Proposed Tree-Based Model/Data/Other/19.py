def bubbleSort(array):
    swap = True
    while swap:
        swap = False
        for x in range(len(array) - 1):
            if array[x] > array[x + 1]:
                array[x], array[x + 1] = array[x + 1], array[x]
                swap = True
