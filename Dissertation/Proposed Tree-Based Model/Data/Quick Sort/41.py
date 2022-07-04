def sort( a,  ulimit,  llimit) :
    quicksort(a, 0, len(a), ulimit, llimit)


def quicksort( a,  start,  stop,  ulimit,  llimit) :
    if (stop - start > 1) :
        p = pivot(a, start, stop, ulimit, llimit)
        quicksort(a, start, p, a[p], llimit)
        quicksort(a, p + 1, stop, ulimit, a[p])


def  pivot( a,  start,  stop,  ulimit,  llimit) :
    p = partition(a, a[start], start + 1, stop, ulimit, llimit)
    if (start < p) :
        swap(a, start, p)
    return p


def  partition( a,  pivot,  start,  stop,  ulimit,  llimit) :
    if (start >= stop) :
        return start - 1
    if (a[start] < pivot) :
        return partition(a, pivot, start + 1, stop, ulimit, llimit)
    if (a[stop - 1] > pivot) :
        return partition(a, pivot, start, stop, ulimit, llimit)
    if (start < stop) :
        swap(a, start, stop)
        return partition(a, pivot, start + 1, stop, ulimit, llimit)
    else :
        return start


def swap( a,  i,  j) :
    temp = a[i]
    a[i] = a[j]
    a[j] = temp