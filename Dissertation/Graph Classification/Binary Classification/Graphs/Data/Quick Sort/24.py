def sort( a) :
    newSort(a, 0, len(a) - 1)


def newSort( a,  lo,  hi) :
    if (hi <= lo) :
        return
    j = partition(a, lo, hi)
    sort(a, lo, j - 1)
    sort(a, j + 1, hi)

def  partition( a,  lo,  hi) :
    i = lo
    j = hi + 1
    v = a[lo]
    while (True) :
        while (less(a[i + 1], v)) :
            if (i == hi) :
                break
        while (less(v, a[j - 1])) :
            if (j == lo) :
                break
        if (i >= j) :
            break
        exch(a, i, j)
    exch(a, lo, j)
    return j

def  select( a,  k) :
    lo = 0
    hi = len(a) - 1
    while (hi > lo) :
        i = partition(a, lo, hi)
        if (i > k) :
            hi = i - 1
        elif(i < k) :
            lo = i + 1
        else :
            return a[i]
    return a[lo]

def  less( v,  w) :
    return v < w

def exch( a,  i,  j) :
    swap = a[i]
    a[i] = a[j]
    a[j] = swap