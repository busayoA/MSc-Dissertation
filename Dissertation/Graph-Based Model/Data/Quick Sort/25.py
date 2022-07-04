def partition(quickSortArray,  front,  rear) :
        pivot = int((front + rear) / 2)
        i = front
        j = rear
        while (i < j) :
            while (quickSortArray[i] <= quickSortArray[pivot] and i < pivot) :
                i += 1
            while (quickSortArray[j] >= quickSortArray[pivot] and j > pivot) :
                j -= 1
            if (quickSortArray[i] > quickSortArray[pivot] and j > pivot) :
                temp = quickSortArray[i]
                quickSortArray[i] = quickSortArray[j]
                quickSortArray[j] = temp
            elif(quickSortArray[i] > quickSortArray[pivot] and j == pivot) :
                temp = quickSortArray[pivot]
                quickSortArray[pivot] = quickSortArray[i]
                quickSortArray[i] = quickSortArray[pivot - 1]
                quickSortArray[pivot - 1] = temp
                pivot -= 1
            if (quickSortArray[j] < quickSortArray[pivot] and i < pivot) :
                temp = quickSortArray[j]
                quickSortArray[j] = quickSortArray[i]
                quickSortArray[i] = temp
            elif(quickSortArray[j] < quickSortArray[pivot] and i == pivot) :
                temp = quickSortArray[pivot]
                quickSortArray[pivot] = quickSortArray[j]
                quickSortArray[j] = quickSortArray[pivot + 1]
                quickSortArray[pivot + 1] = temp
                pivot += 1
        if ((pivot - 1) - front > 0) :
            partition(quickSortArray, front, pivot - 1)
        if (rear - (pivot + 1) > 0) :
            partition(quickSortArray, pivot + 1, rear)