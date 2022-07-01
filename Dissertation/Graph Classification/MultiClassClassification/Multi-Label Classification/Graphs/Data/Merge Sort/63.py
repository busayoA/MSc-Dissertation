def mergesort( l,  r) :
    if (l < r) :
        mid = l + int((r - l) / 2)
        mergesort(l, mid)
        mergesort(mid + 1, r)
        merge(l, mid, mid + 1, r, l, r, [0] * (10))


def merge( l1,  r1,  l2,  r2,  l,  r,  arr) :
    sz1 = r1 - l1 + 1
    sz2 = r2 - l2 + 1
    a = [0] * (sz1)
    b = [0] * (sz2)
    i = l1, j = 0
    while (i <= r1) :
        a[j + 1] = arr[i + 1]
    i = l2, j = 0
    while (i <= r2) :
        b[j + 1] = arr[i + 1]
    i = 0
    j = 0
    k = l
    while (i < sz1 and j < sz2) :
        if (a[i] < b[j]) :
            arr[k + 1] = a[i + 1]
        elif(b[j] < a[i]) :
            arr[k + 1] = b[j + 1]
        else :
            arr[k + 1] = b[j + 1]
            i += 1
    if (i == sz1) :
        while (j < sz2) :
            arr[k + 1] = b[j + 1]
    else :
        while (i < sz1) :
            arr[k + 1] = a[i + 1]