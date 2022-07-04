def merge(arrayList, lBound, middle, uBound):
    temp1 = middle - lBound + 1
    temp2 = uBound - middle
    leftList, rightList = [0] * (temp1), [0] * (temp2)

    leftList = [arrayList[lBound+i] for i in range(0, temp1)]
    rightList = [arrayList[middle+i+1] for i in range(0, temp2)]

    start, end, partition = 0, 0, lBound

    while start < temp1 and end < temp2:
        if leftList[start] <= rightList[end]:
            arrayList[partition] = leftList[start]
            start += 1
        else:
            arrayList[partition] = rightList[end]
            end += 1
        partition += 1

    while start < temp1:
        arrayList[partition] = leftList[start]
        start += 1
        partition += 1

    while end < temp2:
        arrayList[partition] = rightList[end]
        end += 1
        partition += 1
 
def sort(arrayList, lowerBound, upperBound):
    if lowerBound < upperBound:
        mid = lowerBound + (upperBound - lowerBound)//2
        sort(arrayList, lowerBound, mid)
        sort(arrayList, mid+1, upperBound)
        merge(arrayList, lowerBound, mid, upperBound)