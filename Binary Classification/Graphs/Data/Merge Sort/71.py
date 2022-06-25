temp = []
def sort(a) :
    n = len(a)
    temp = [0.0] * (n)
    recursiveSort(a, 0, n - 1)

def recursiveSort(a,  fromm,  to) :
    if (to - fromm < 2) :
        if (to > fromm and a[to] < a[fromm]) :
            aTemp = a[to]
            a[to] = a[fromm]
            a[fromm] = aTemp
    else :
        middle = int((fromm + to) / 2)
        recursiveSort(a, fromm, middle)
        recursiveSort(a, middle + 1, to)
        merge(a, fromm, middle, to)

def merge( a,  fromm,  middle,  to) :
    i = fromm
    j = middle + 1
    k = fromm
    while (i <= middle and j <= to) :
        if (a[i] < a[j]) :
            temp[k] = a[i]
            i += 1
        else :
            temp[k] = a[j]
            j += 1
        k += 1
    while (i <= middle) :
        temp[k] = a[i]
        i += 1
        k += 1
    while (j <= to) :
        temp[k] = a[j]
        j += 1
        k += 1
    k = fromm
    while (k <= to) :
        a[k] = temp[k]
        k += 1