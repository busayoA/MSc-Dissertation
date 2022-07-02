def quickSortHelper(num,  left,  right) :
    if (left >= right) :
        return
    pivot = partition(num, left, right)
    quickSortHelper(num, left, pivot - 1)
    quickSortHelper(num, pivot + 1, right)

def partition(num,  left,  right) :
    pivot = right
    print("pivot: " + str(pivot), end ="")
    i = left
    print(" left: " + str(num[i]), end ="")
    while (i < pivot) :
        while (i < pivot and num[pivot] >= num[i]) :
            i += 1
        if (i < pivot) :
            swap(num, i, pivot - 1)
            swap(num, pivot, pivot - 1)
            pivot -= 1
    return pivot

def swap(num,  i,  j) :
    tmp = num[i]
    num[i] = num[j]
    num[j] = tmp