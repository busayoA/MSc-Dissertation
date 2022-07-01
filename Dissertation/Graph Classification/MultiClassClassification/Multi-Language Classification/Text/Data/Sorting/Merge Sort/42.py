def  mergeSort(self, num) :
        if (len(num) == 0 or len(num) == 1) :
            return num
        left = 0
        right = len(num) - 1
        mergeSortHelper(num, left, right)
        print("[", end ="")
        i = 0
        while (i < len(num)) :
            print(str(num[i]) + ",", end ="")
            i += 1
        print("]", end ="")
        return num

def mergeSortHelper(self, num,  left,  right) :
    if (left < right) :
        middle = left + int((right - left) / 2)
        mergeSortHelper(num, left, middle)
        mergeSortHelper(num, middle + 1, right)
        merge(num, left, middle, right)

def merge(self, num,  left,  middle,  right) :
    tmp = [0] * (right + 1 - left)
    j = middle + 1
    i = left
    k = 0
    while (i <= middle and j <= right) :
        if (num[i] < num[j]) :
            tmp[k] = num[i]
            i += 1
        else :
            tmp[k] = num[j]
            j += 1
        k += 1
    while (k < len(tmp)) :
        if (i <= middle) :
            tmp[k] = num[i]
            i += 1
        if (j <= right) :
            tmp[k] = num[j]
            j += 1
        k += 1
    e = 0
    while (e < len(tmp)) :
        num[left + e] = tmp[e]
        e += 1