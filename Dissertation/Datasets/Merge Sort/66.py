def merge( a,  aux,  lo,  mid,  hi) :
    k = lo
    while (k <= hi) :
        aux[k] = a[k]
        k += 1
    i = lo
    j = mid + 1
    k = lo
    while (k <= hi) :
        if (i > mid) :
            a[k] = aux[j + 1]
        elif(j > hi) :
            a[k] = aux[i + 1]
        else :
            a[k] = aux[i + 1]
        k += 1

def sort( a,  aux,  lo,  hi) :
    if (hi <= lo) :
        return
    mid = lo + int((hi - lo) / 2)
    sort(a, aux, lo, mid)
    sort(a, aux, mid + 1, hi)
    merge(a, aux, lo, mid, hi)


def sort( a) :
    aux = [None] * (len(a))
    sort(a, aux, 0, len(a) - 1)