def sort(myList, start, end):
    if start < end:
        middle = start + (end - start) // 2
        sort(myList, start, middle)
        sort(myList, middle + 1, end)
        merge(myList, start, middle, end)

def merge(myList, start, middle, end):
    leftList = [(myList[start+i]) for i in range(0, middle-start+1)]
    rightList = [myList[middle+i+1] for i in range(0, end-middle)]

    left, right = 0, 0
    for k in range(start, end+1):
        if left == len(leftList):
            myList[k] = rightList[right]
            right += 1
        elif right == len(rightList):
            myList[k] = leftList[left]
            left += 1
        elif leftList[left] <= rightList[right]:
            myList[k] = leftList[left]
            left += 1
        else:
            myList[k] = rightList[right]
            right += 1