def quicksort( a,  start,  end) :
        if (start >= end) :
            return
        pavot = a[start]
        i = start
        j = end
        while (i < j) :
            while (i < j and a[j] > pavot) :
                j -= 1
            while (i < j and a[i] <= pavot) :
                i += 1
            if (i != j) :
                t = a[i]
                a[i] = a[j]
                a[j] = t
        if (i == j) :
            t = a[start]
            a[start] = a[i]
            a[j] = t
        quicksort(a, start, i - 1)
        quicksort(a, i + 1, end)