def sort(array,  fromm,  end) :
    quickSort(array, fromm, end)

def quickSort(array,  low,  high) :
    if (low < high) :
        pivot = partition2(array, low, high)
        quickSort(array, low, pivot - 1)
        quickSort(array, pivot + 1, high)

def swap(self, array,  i,  j) :
    if (i != j) :
        tmp = array[i]
        array[i] = array[j]
        array[j] = tmp

def  partition2(self, array,  low,  high) :
    pivot = low
    while (True) :
        if (pivot != high) :
            if ((array[high].compareTo(array[pivot])) < 0) :
                self.swap(array, high, pivot)
                pivot = high
            else :
                high -= 1
        else :
            if ((array[low].compareTo(array[pivot])) > 0) :
                self.swap(array, low, pivot)
                pivot = low
            else :
                low += 1
        if (low == high) :
            break
    return pivot