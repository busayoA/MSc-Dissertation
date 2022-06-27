def quickSortInPlace( S,  comp) :
    quickSortInPlace2(S, comp, 0, len(S) - 1)


def quickSortInPlace2( S,  comp,  a,  b) :
    if (a >= b) :
        return
    left = a
    right = b - 1
    pivot = S[b]
    temp = None
    while (left <= right) :
        while (left <= right and comp.compare(S[left],pivot) < 0) :
            left += 1
        while (left <= right and comp.compare(S[right],pivot) > 0) :
            right -= 1
        if (left <= right) :
            temp = S[left]
            S[left] = S[right]
            S[right] = temp
            left += 1
            right -= 1
    temp = S[left]
    S[left] = S[b]
    S[b] = temp
    quickSortInPlace(S, comp, a, left - 1)
    quickSortInPlace(S, comp, left + 1, b)