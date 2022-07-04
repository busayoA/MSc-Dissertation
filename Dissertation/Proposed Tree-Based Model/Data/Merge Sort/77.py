def sort(a,  aux,  lo,  hi) :
    if (lo >= hi) :
        return
    mid = lo + int((hi - lo) / 2)
    sort(a, aux, lo, mid)
    sort(a, aux, mid + 1, hi)
    merge(a, aux, lo, mid, hi)

def merge(a,  aux,  lo,  mid,  hi) :
    i = lo
    while (i <= hi) :
        aux[i] = a[i]
        i += 1
    firstHalf = lo
    secondHalf = mid + 1
    i = lo
    while (i <= hi) :
        if (firstHalf > mid) :
            a[i] = aux[secondHalf + 1]
        elif(secondHalf > hi) :
            a[i] = aux[firstHalf + 1]
        elif(aux[secondHalf] < aux[firstHalf]) :
            a[i] = aux[secondHalf + 1]
        else :
            a[i] = aux[firstHalf + 1]
        i += 1

def sort(a) :
    aux = [0] * (len(a))
    sort(a, aux, 0, len(a) - 1)