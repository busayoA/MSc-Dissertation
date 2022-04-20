import random
def quickSortAlgWithRd(numbers,  left,  right) :
    if (left < right) :
        index = int((random() * (right - left + 1) + left))
        flag = numbers[index]
        i = left
        j = right
        k = index
        while (i < j) :
            while (j > i and numbers[j] > flag) :
                j -= 1
            while (i < j and numbers[i] <= flag) :
                i += 1
            numbers[k] = numbers[j]
            numbers[j] = numbers[i]
            k = i
        numbers[k] = flag
        quickSortAlgWithRd(numbers, left, k - 1)
        quickSortAlgWithRd(numbers, k + 1, right)

def quickSortAlgFromLeft(numbers,  left,  right) :
    if (left < right) :
        flag = numbers[left]
        i = left
        j = right
        while (i < j) :
            while (j > i and numbers[j] > flag) :
                j -= 1
            if (j > i) :
                numbers[i + 1] = numbers[j]
            while (i < j and numbers[i] <= flag) :
                i += 1
            if (i < j) :
                numbers[j - 1] = numbers[i]
        numbers[i] = flag
        quickSortAlgFromLeft(numbers, left, i - 1)
        quickSortAlgFromLeft(numbers, i + 1, right)

def quickSortAlgFromRight(numbers,  left,  right) :
    if (left < right) :
        flag = numbers[right]
        i = left
        j = right
        while (i < j) :
            while (i < j and numbers[i] <= flag) :
                i += 1
            if (i < j) :
                numbers[j - 1] = numbers[i]
            while (j > i and numbers[j] > flag) :
                j -= 1
            if (j > i) :
                numbers[i +1] = numbers[j]
        numbers[j] = flag
        quickSortAlgFromRight(numbers, left, j - 1)
        quickSortAlgFromRight(numbers, j + 1, right)