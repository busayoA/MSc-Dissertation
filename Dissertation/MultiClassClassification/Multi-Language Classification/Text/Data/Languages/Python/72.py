def mergeSort( A) :
    mergeSort(A, 0, len(A))

def mergeSort( A,  p,  r) :
    if (p < r) :
        q = (int((p + r) / 2))
        mergeSort(A, p, q)
        mergeSort(A, q + 1, r)
        merge(A, p, q, r)

def merge( A,  p,  q,  r) :
    n1 = q - p
    n2 = r - q
    L = [0] * (n1 + 1)
    R = [0] * (n2 + 1)
    i = 0
    while (i < n1) :
        L[i] = A[p + i]
        i += 1
    j = 0
    while (j < n2) :
        R[j] = A[q + j]
        j += 1
    L[n1] = []
    R[n2] = []
    i = 0
    j = 0
    k = p
    while (k < r) :
        if (L[i] <= R[j]) :
            A[k] = L[i]
            i += 1
        else :
            A[k] = R[j]
            j += 1
        k += 1