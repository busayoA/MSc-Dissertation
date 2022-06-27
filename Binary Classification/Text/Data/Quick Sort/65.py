def sort( a,  low,  hight) :
    i = 0
    j = 0
    index = 0
    if (low > hight) :
        return
    i = low
    j = hight
    index = a[i]
    loop = 0
    while (i < j) :
        while (i < j and a[j] >= index) :
            j -= 1
        if (i < j) :
            a[i + 1] = a[j]
        while (i < j and a[i] < index) :
            i += 1
        if (i < j) :
            a[j - 1] = a[i]
        print("i is " + str(i))
        print("j is " + str(j))
        loop += 1
        print(str(loop) + " loop " + str(a))
    print("===")
    a[i] = index
    sort(a, low, i - 1)
    sort(a, i + 1, hight)


def quickSort( a) :
    sort(a, 0, len(a) - 1)