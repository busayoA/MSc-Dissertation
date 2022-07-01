def quickSort( ar) :
    n = len(ar)
    left =  []
    right =  []
    value = ar[0]
    l = 0
    r = 0
    c = 1
    i = 1
    while (i < n) :
        if (ar[i] > value) :
            right.append(ar[i])
            r += 1
        elif(ar[i] < value) :
            left.append(ar[i])
            l += 1
        else :
            c += 1
        i += 1
    if (len(left) > 1) :
        quickSort(left)
        i = 0
        while (i < l) :
            print(str(left[i]) + " ", end ="")
            i += 1
        print()
    if (len(right) > 1) :
        quickSort(right)
        i = 0
        while (i < r) :
            print(str(right[i]) + " ", end ="")
            i += 1
        print()
    count = 0
    i = 0
    while (i < l) :
        ar[count + 1] = left[i]
        i += 1
    i = 0
    while (i < c) :
        ar[count + 1] = value
        i += 1
    i = 0
    while (i < r) :
        ar[count + 1] = right[i]
        i += 1
    return