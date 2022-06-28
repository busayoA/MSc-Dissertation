def  countInversions( arr) :
        aux = arr.clone()
        return countInversions(arr, 0, len(arr) - 1, aux)

def  countInversions( arr,  lo,  hi,  aux) :
    if (lo >= hi) :
        return 0
    mid = lo + int((hi - lo) / 2)
    count = 0
    count += countInversions(aux, lo, mid, arr)
    count += countInversions(aux, mid + 1, hi, arr)
    count += merge(arr, lo, mid, hi, aux)
    return count

def  merge( arr,  lo,  mid,  hi,  aux) :
    count = 0
    i = lo
    j = mid + 1
    k = lo
    while (i <= mid or j <= hi) :
        if (i > mid) :
            arr[k + 1] = aux[j + 1]
        elif(j > hi) :
            arr[k + 1] = aux[i + 1]
        elif(aux[i] <= aux[j]) :
            arr[k + 1] = aux[i +1]
        else :
            arr[k + 1] = aux[j + 1]
            count += mid + 1 - i
    return count