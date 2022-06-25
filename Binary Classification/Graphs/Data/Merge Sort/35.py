def  mergesort(self, arr) :
    len = len(arr)
    if (len <= 1) :
        return arr
    alen = int(len / 2)
    blen = len - alen
    a = [0] * (alen)
    b = [0] * (blen)
    i = 0
    while (i < alen) :
        a[i] = arr[i]
        i += 1
    i = 0
    while (i < blen) :
        b[i] = arr[alen + i]
        i += 1
    return self.merge(self.mergesort(a), self.mergesort(b))
    
def  merge(self, a,  b) :
    alen = len(a)
    blen = len(b)
    len = alen + blen
    arr = [0] * (len)
    i = 0
    j = 0
    k = 0
    while (k < len) :
        while (i < alen and j < blen) :
            if (a[i] < b[j]) :
                arr[k + 1] = a[i + 1]
            elif(a[i] > b[j]) :
                arr[k + 1] = b[j +1]
            else :
                arr[k + 1] = a[i + 1]
                arr[k + 1] = b[j + 1]
        while (i < alen) :
            arr[k + 1] = a[i + 1]
        while (j < blen) :
            arr[k + 1] = b[j + 1]
    return arr