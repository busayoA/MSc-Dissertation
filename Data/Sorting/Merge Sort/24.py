def mergeSort(self, arrayToSort,  low,  high) :
    if (high - low <= 1) :
        return
    mid = low + int((high - low) / 2)
    part = [None] * (high - low)
    self.mergeSort(arrayToSort, low, mid)
    self.mergeSort(arrayToSort, mid, high)
    self.merge(arrayToSort, low, mid, high, part)
        
def  merge(self, arrayToSort,  low,  mid,  high,  part) :
    first = low
    second = mid
    mergeIndex = 0
    while (first < mid and second < high) :
        if (arrayToSort[first] <= arrayToSort[second]) :
            part[mergeIndex] = arrayToSort[first]
            first += 1
        else :
            part[mergeIndex] = arrayToSort[second]
            second += 1
        mergeIndex += 1
    while (first < mid) :
        part[mergeIndex] = arrayToSort[first]
        first += 1
        mergeIndex += 1
    while (second < high) :
        part[mergeIndex] = arrayToSort[second]
        second += 1
        mergeIndex += 1
    mergeIndex = 0
    count = low
    while (count < high) :
        arrayToSort[count] = part[mergeIndex]
        mergeIndex += 1
        count += 1
    return part