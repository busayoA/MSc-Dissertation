def sort( a) :
    sort2(a, 0, len(a) - 1)


def sort2( a,  lo,  hi) :
    if (hi <= lo) :
        return
    j = partition(a, lo, hi)
    sort(a, lo, j - 1)
    sort(a, j + 1, hi)


def  partition( a,  lo,  hi) :
    i = lo
    j = hi + 1
    while (True) :
        while (a[lo] > a[i + 1]) :
            if (i == hi) :
                break
        while (a[lo] < a[j - 1]) :
            if (j == lo) :
                break
        if (j <= i) :
            break
        exch(a, i, j)
    exch(a, lo, j)
    return j


def exch( a,  i,  j) :
    temp = a[i]
    a[i] = a[j]
    a[j] = temp