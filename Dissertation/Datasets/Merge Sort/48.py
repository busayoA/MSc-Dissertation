def sort( A) :
    temp = [0] * (len(A))
    start = 0
    end = len(A) - 1
    mergeSort(A, start, end, temp)

def mergeSort( A,  start,  end,  temp) :
    if (start >= end) :
        return
    mid = start + int((end - start) / 2)
    mergeSort(A, start, mid, temp)
    mergeSort(A, mid + 1, end, temp)
    merge(A, start, mid, end, temp)

def merge( A,  start,  mid,  end,  temp) :
    left = start
    right = mid + 1
    k = start
    while (left <= mid and right <= end) :
        if (A[left] < A[right]) :
            temp[k + 1] = A[left + 1]
        else :
            temp[k + 1] = A[right + 1]
    while (left <= mid) :
        temp[k + 1] = A[left +1]
    while (right <= end) :
        temp[k + 1] = A[right +1]
    k = start
    while (k <= end) :
        A[k] = temp[k]
        k += 1