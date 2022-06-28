def mergesort( ar,  i,  f) :
    if (i < f) :
        m = int((i + f) / 2)
        mergesort(ar, i, m)
        mergesort(ar, m + 1, f)
        merge(ar, i, f, m)

def merge( ar,  i,  f,  m) :
    aux = ([None] * (f - i + 1))
    k = 0
    a = i
    b = m + 1
    while (a <= m and b <= f) :
        if (ar[a] < ar[b]) :
            aux[k] = ar[a]
            k += 1
            a += 1
        else :
            aux[k] = ar[b]
            k += 1
            b += 1
    while (a <= m) :
        aux[k] = ar[a]
        k += 1
        a += 1
    while (b <= f) :
        aux[k] = ar[b]
        k += 1
        b += 1
    a = i
    k = 0
    while (a <= f) :
        ar[a] = aux[k]
        a += 1
        k += 1