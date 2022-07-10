def mergeSort(self, A,  start,  end,  temp) :
    if (start < end) :
        mid = start + int((end - start) / 2)
        mergeSort(A, start, mid, temp)
        mergeSort(A, mid + 1, end, temp)
        merge2(A, start, mid, end, temp)

def merge2(self, A,  start,  mid,  end,  temp) :
    i = mid
    j = end
    k = end
    while (i >= start and j >= mid + 1) :
        if (A[i] > A[j]) :
            temp[k - 1] = A[i - 1]
        else :
            temp[k - 1] = A[j -1]
    if (i < start) :
        while (j >= mid + 1) :
            temp[k - 1] = A[j - 1]
    if (j < mid + 1) :
        while (i >= start) :
            temp[k - 1] = A[i - 1]
    p = start
    while (p <= end) :
        A[p] = temp[p]
        p += 1