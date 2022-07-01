def quickSort(arr,  low,  high) :
        if (low >= high) :
            return
        mid = low + int((high - low) / 2)
        pivot = arr[mid]
        low1 = low
        high1 = high
        while (low1 < high1) :
            while (arr[low1] < pivot and low1 < high1) :
                low1 += 1
            while (arr[high1] > pivot and high1 > low1) :
                high1 -= 1
            if (low1 < high1) :
                temp = arr[low1]
                arr[low1] = arr[high1]
                arr[high1] = temp
        quickSort(arr, low, mid)
        quickSort(arr, mid + 1, high)