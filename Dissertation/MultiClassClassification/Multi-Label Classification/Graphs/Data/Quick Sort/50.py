def quicksort( a,  lo,  hi) :
        i = lo
        j = hi
        h = 0
        x = a[int((lo + hi) / 2)]
        while True :
            while (a[i] < x) :
                i += 1
            while (a[j] > x) :
                j -= 1
            if (i <= j) :
                h = a[i]
                a[i] = a[j]
                a[j] = h
                i += 1
                j -= 1
            if((i <= j) == False) :
                    break
        if (lo < j) :
            quicksort(a, lo, j)
        if (i < hi) :
            quicksort(a, i, hi)