def mergesort0(num):
    mergesort(num, 0, len(num)-1)
    return (num)

def mergesort( num, low, high):
    if low < high:
        mid = (low + high) // 2
        mergesort(num, low, mid)
        mergesort(num, mid+1, high)
        merge(num, low, mid, mid+1, high)

def merge(a, l1, u1, l2, u2):
    temp = [0]*len(a)
    i = l1
    j = l2
    k = l1
    while (i <= u1 and j <= u2):
        if (a[i] <= a[j]):
            temp[k] = a[i]
            i = i + 1
        else:
            temp[k] = a[j]
            j = j + 1

        k = k + 1
    while ( i <= u1 ):
        temp[k] = a[i]
        k = k + 1
        i = i + 1
    while ( j <= u2 ):
        temp[k] = a[j]
        k = k + 1
        j = j + 1

    h = l1

    while ( h <= u2 ):
        a[h] = temp[h]
        h = h + 1