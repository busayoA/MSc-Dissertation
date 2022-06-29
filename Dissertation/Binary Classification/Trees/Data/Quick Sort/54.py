def quicksort(A,  p,  r) :
    if (p < r) :
        q = partition(A, p, r)
        quicksort(A, p, q - 1)
        quicksort(A, q + 1, r)

def  partition(A,  p,  r) :
    x = A[r]
    i = p - 1
    j = p
    while (j < r) :
        if (A[j] <= x) :
            i = i + 1
            swap(A, i, j)
        swap(A, i + 1, r)
        j += 1
    return i + 1

def swap(A,  i,  j) :
    temp = A[i]
    A[i] = A[j]
    A[j] = temp