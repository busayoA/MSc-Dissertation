def mergeSort(self, A,  start,  end,  temp) :
        if (start >= end) :
            return
        left = start
        right = end
        mid = int((start + end) / 2)
        mergeSort(A, start, mid, temp)
        mergeSort(A, mid + 1, end, temp)
        merge(A, start, mid, end, temp)

def merge(self, A,  start,  mid,  end,  temp) :
    left = start
    right = mid + 1
    index = start
    while (left <= mid and right <= end) :
        if (A[left] < A[right]) :
            temp[index + 1] = A[left + 1]
        else :
            temp[index + 1] = A[right +1]
    while (left <= mid) :
        temp[index + 1] = A[left + 1]
    while (right <= end) :
        temp[index + 1] = A[right + 1]
    index = start
    while (index <= end) :
        A[index] = temp[index]
        index += 1