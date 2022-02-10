def merge(array, lowerBound, middle, upperBound):
    leftList = [array[lowerBound + i] for i in range(middle - lowerBound + 1)]
    rightList = [array[middle + i + 1] for i in range(upperBound - middle)]

    left, right, partition = 0, 0, lowerBound
    while left < len(leftList) and right < len(rightList):
        if leftList[left] <= rightList[right]:
            array[partition] = leftList[left]
            left += 1
        else:
            array[partition] = rightList[right]
            right += 1
        partition += 1

    while left < len(leftList):
        array[partition] = leftList[left]
        left += 1
        partition += 1
    
    while right < len(rightList):
        array[partition] = rightList[right]
        right += 1
        partition += 1

def sort(array, lowerBound, upperBound):
    if (lowerBound < upperBound):
        middle = (lowerBound + upperBound) // 2
        sort(array, lowerBound, middle)
        sort(array, middle + 1, upperBound)
        merge(array, lowerBound, middle, upperBound)
