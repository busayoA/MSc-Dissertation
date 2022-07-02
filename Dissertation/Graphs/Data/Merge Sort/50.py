def  mergeSort(self, a) :
        if (len(a) <= 1) :
            return a
        middle = int(len(a) / 2)
        return merge(mergeSort(a[0:middle]), mergeSort(a[middle:len(a)]))

def  merge(self, a1,  a2) :
    a = 0
    b = 0
    arr = [0] * (len(a1) + len(a2))
    i = 0
    while (i < len(a1) + len(a2)) :
        if (a < len(a1) and b < len(a2)) :
            if (a1[a] <= a2[b]) :
                arr[i] = a1[a + 1]
            else :
                arr[i] = a2[b + 1]
        elif(a < len(a1)) :
            arr[i] = a1[a + 1]
        else :
            arr[i] = a2[b +1]
        i += 1
    return arr