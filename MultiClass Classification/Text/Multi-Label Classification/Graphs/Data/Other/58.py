def insertionSort(array):
    for i in range(1, len(array)): #we start at index 1 because index 0 is technically already sorted
        j = i
        while j>0 and array[j] < array[j-1]: #while j hasn't reached the very beginning of the array and while there are still numbers left to be sorted
            swap(j, j-1, array) #swap current J with the index before it
            j-=1    #decrement J to keep track of the number we are trying to insert
    return array

def swap(i, j, array):
    array[i], array[j] = array[j], array[i]