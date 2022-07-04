def sort( a) :
    sort(a, 0, len(a))

def sort( a,  lo,  hi) :
    N = hi - lo
    if (N <= 1) :
        return
    mid = lo + int(N / 2)
    sort(a, lo, mid)
    sort(a, mid, hi)
    aux = [None] * (N)
    i = lo
    j = mid
    k = 0
    while (k < N) :
        if (i == mid) :
            aux[k] = a[j + 1]
        elif(j == hi) :
            aux[k] = a[i + 1]
        elif(a[j].compareTo(a[i]) < 0) :
            aux[k] = a[j + 1]
        else :
            aux[k] = a[i + 1]
        k += 1
    k = 0
    while (k < N) :
        a[lo + k] = aux[k]
        k += 1