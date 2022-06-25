def mergeSort( A) :
        if (A == None or len(A) == 0) :
            return
        helper = A[len(A):]
        divideAndMerge(A, 0, len(A) - 1, helper)

def divideAndMerge( A,  l,  r,  helper) :
    if (l >= r) :
        return
    mid = int((l + r) / 2)
    divideAndMerge(A, l, mid, helper)
    divideAndMerge(A, mid + 1, r, helper)
    merge(A, l, mid, r, helper)

def merge( A,  l,  mid,  r,  helper) :
    ind1 = l
    ind2 = mid + 1
    ind = l
    while (ind1 <= mid and ind2 <= r) :
        if (helper[ind1] < helper[ind2]) :
            A[ind + 1] = helper[ind1 + 1]
        else :
            A[ind + 1] = helper[ind2 + 1]
    while (ind1 <= mid) :
        A[ind + 1] = helper[ind1 + 1]
    i = l
    while (i <= r) :
        helper[i] = A[i]
        i += 1