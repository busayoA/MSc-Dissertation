def sort( A) :
    quick_sort(A, 0, len(A) - 1)


def  partition( A,  p,  r) :
    x = A[r]
    tmp = 0
    i = p - 1
    j = p
    while (j < r) :
        if (A[j] <= x) :
            i += 1
            tmp = A[j]
            A[j] = A[i]
            A[i] = tmp
        j += 1
    i += 1
    A[r] = A[i]
    A[i] = x
    return i


def quick_sort( A,  p,  r) :
    if (p < r) :
        q = partition(A, p, r)
        quick_sort(A, p, q - 1)
        quick_sort(A, q + 1, r)