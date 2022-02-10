def bSort(array):
    swap = True
    while swap is True:
        swap = False
        for i in range(len(array)-1):
            if array[i] > array[i+1]:
                swap = True
                temp = array[i]
                array[i] = array[i+1]
                array[i+1] = temp
