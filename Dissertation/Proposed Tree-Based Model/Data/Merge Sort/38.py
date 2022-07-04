def merge(self, A,  l,  m,  r) :
    nL = m - l + 1
    nR = r - m
    left = [0] * (nL)
    right = [0] * (nR)
    i = 0
    while (i < nL) :
        left[i] = A[l + i]
        i += 1
    j = 0
    while (j < nR) :
        right[j] = A[m + 1 + j]
        j += 1
    i = 0
    j = 0
    k = l
    while (i < nL and j < nR) :
        if (left[i] <= right[j]) :
            A[k] = left[i]
            i += 1
        else :
            A[k] = right[j]
            j += 1
        k += 1
    while (i < nL) :
        A[k] = left[i]
        i += 1
        k += 1
    while (j < nR) :
        A[k] = right[j]
        j += 1
        k += 1

def merge_sort(self, A,  l,  r) :
    if (l < r) :
        mid = int((l + r) / 2)
        merge_sort(A, l, mid)
        merge_sort(A, mid + 1, r)
        merge(A, l, mid, r)