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
            aux[k] = a[j +1]
        elif(j == hi) :
            aux[k] = a[i +1]
        elif(a[j].compareTo(a[i]) < 0) :
            aux[k] = a[j +1]
        else :
            aux[k] = a[i + 1]
        k += 1
    k = 0
    while (k < N) :
        a[lo + k] = aux[k]
        k += 1

def  isSorted( a) :
    i = 1
    while (i < len(a)) :
        if (a[i].compareTo(a[i - 1]) < 0) :
            return False
        i += 1
    return True

def show( a) :
    i = 0
    while (i < len(a)) :
        print(a[i])
        i += 1