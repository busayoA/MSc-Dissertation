def swap( a,  l,  m) :
    temp = a[l]
    a[l] = a[m]
    a[m] = temp


def  partition( a,  lo,  hi) :
    pivot = a[lo]
    j = lo
    i = lo + 1
    while (i <= hi) :
        if (a[i] < pivot) :
            swap(a, i, j +1)
        i += 1
    swap(a, j, lo)
    return j


def sort( a) :
    q_sort3way(a, 0, len(a) - 1)


def q_sort3way( a,  lo,  hi) :
    if (lo >= hi) :
        return
    i = lo + 1
    l = lo
    m = hi
    pivot = a[lo]
    cmp = 0
    while (i <= m) :
        cmp = (a[i] > pivot)
        if (cmp == 0) :
            i += 1
        elif(cmp < 0) :
            swap(a, i + 1, l +1)
        else :
            swap(a, i, m - 1)
    q_sort3way(a, lo, l - 1)
    q_sort3way(a, m + 1, hi)


def sort( a,  lo,  hi) :
    if (lo >= hi) :
        return
    p = partition(a, lo, hi)
    sort(a, lo, p - 1)
    sort(a, p + 1, hi)

def  partitionGlobal(a,  lo,  hi) :
    if (a == None) :
        return -1
    if (lo == hi) :
        return lo
    pivot = a[lo]
    i = lo
    j = lo + 1
    while (j <= hi) :
        if (a[j] < pivot) :
            swap(a, i, j)
        j += 1
    swap(a, lo, i)
    return i