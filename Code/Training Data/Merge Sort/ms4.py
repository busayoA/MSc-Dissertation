def merge(myList, lowerBound, middle, upperBound):
    leftList = [myList[lowerBound + i] for i in range(middle - lowerBound + 1)]
    rightList = [myList[middle + i + 1] for i in range(upperBound - middle)]

    left, right, partition = 0, 0, lowerBound
    while left < len(leftList) and right < len(rightList):
        if leftList[left] <= rightList[right]:
            myList[partition] = leftList[left]
            left += 1
        else:
            myList[partition] = rightList[right]
            right += 1
        partition += 1

    while left < len(leftList):
        myList[partition] = leftList[left]
        left += 1
        partition += 1
    
    while right < len(rightList):
        myList[partition] = rightList[right]
        right += 1
        partition += 1

def sort(myList, lowerBound, upperBound):
    if (lowerBound < upperBound):
        middle = (lowerBound + upperBound) // 2
        sort(myList, lowerBound, middle)
        sort(myList, middle + 1, upperBound)
        merge(myList, lowerBound, middle, upperBound)
