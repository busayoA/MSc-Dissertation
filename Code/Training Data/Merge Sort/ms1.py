def merge(myList, lBound, middle, uBound):
    temp1 = middle - lBound + 1
    temp2 = uBound - middle
    leftList, rightList = [0] * (temp1), [0] * (temp2)

    leftList = [myList[lBound+i] for i in range(0, temp1)]
    rightList = [myList[middle+i+1] for i in range(0, temp2)]

    start, end, partition = 0, 0, lBound

    while start < temp1 and end < temp2:
        if leftList[start] <= rightList[end]:
            myList[partition] = leftList[start]
            start += 1
        else:
            myList[partition] = rightList[end]
            end += 1
        partition += 1

    while start < temp1:
        myList[partition] = leftList[start]
        start += 1
        partition += 1

    while end < temp2:
        myList[partition] = rightList[end]
        end += 1
        partition += 1
 
def sort(myList, lowerBound, upperBound):
    if lowerBound < upperBound:
        mid = lowerBound + (upperBound - lowerBound)//2
        sort(myList, lowerBound, mid)
        sort(myList, mid+1, upperBound)
        merge(myList, lowerBound, mid, upperBound)