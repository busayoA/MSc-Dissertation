def sort(arr, start, end):
    if start < end:
        middle = start + (end - start) // 2
        sort(arr, start, middle)
        sort(arr, middle + 1, end)
        merge(arr, start, middle, end)

def merge(arr, start, middle, end):
    leftList = [(arr[start+i]) for i in range(0, middle-start+1)]
    rightList = [arr[middle+i+1] for i in range(0, end-middle)]

    left, right = 0, 0
    for k in range(start, end+1):
        if left == len(leftList):
            arr[k] = rightList[right]
            right += 1
        elif right == len(rightList):
            arr[k] = leftList[left]
            left += 1
        elif leftList[left] <= rightList[right]:
            arr[k] = leftList[left]
            left += 1
        else:
            arr[k] = rightList[right]
            right += 1