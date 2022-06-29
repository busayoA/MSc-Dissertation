def quickSort( nums,  left,  right) :
        if (left >= right) :
            return
        pivot = nums[left]
        i = left
        j = right
        while (i < j) :
            while (nums[j] >= pivot and i < j) :
                j -= 1
            while (nums[i] <= pivot and i < j) :
                i += 1
            if (i < j) :
                tmp = nums[i]
                nums[i] = nums[j]
                nums[j] = tmp
        nums[left] = nums[i]
        nums[i] = pivot
        quickSort(nums, left, i - 1)
        quickSort(nums, i + 1, right)