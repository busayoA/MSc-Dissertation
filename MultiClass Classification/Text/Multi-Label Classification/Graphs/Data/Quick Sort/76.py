def  partition( nums,  low,  high) :
    pivot = nums[high]
    i = low - 1
    j = low
    while (j < high) :
        if (nums[j] <= pivot) :
            i += 1
            temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
        j += 1
    temp = nums[i + 1]
    nums[i + 1] = nums[high]
    nums[high] = temp
    return i + 1


def quickSort( nums,  low,  high) :
    if (low < high) :
        index = partition(nums, low, high)
        quickSort(nums, low, index - 1)
        quickSort(nums, index + 1, high)