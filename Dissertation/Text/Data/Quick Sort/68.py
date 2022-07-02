def sort( a) :
    sort2(a, 0, len(a) - 1)

def sort2( a,  low,  high) :
    if (low >= high) :
        return
    j = partition(a, low, high)
    sort2(a, low, j - 1)
    sort2(a, j + 1, high)


def  partition( a,  low,  high) :
    N = low
    i = low
    j = high + 1
    while (True) :
        while (a[N] > a[i + 1]) :
            if (i == high) :
                break
        if (i >= j) :
            break
        swap = a[j]
        a[j] = a[i]
        a[i] = swap
    swap = a[j]
    a[j] = a[N]
    a[N] = swap
    return j