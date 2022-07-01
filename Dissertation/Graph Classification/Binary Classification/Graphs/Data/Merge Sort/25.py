def sort(array) :
        tmp = [0] * (len(array))
        sort.mergeSort(array, tmp, 0, len(array) - 1)

def mergeSort( array,  tmp,  left,  right) :
        if (left < right) :
            center = int((left + right) / 2)
            sort.mergeSort(array, tmp, left, center)
            sort.mergeSort(array, tmp, center + 1, right)
            sort.merge(array, tmp, left, center + 1, right)

def merge( array,  tmp,  left,  right,  rightEnd) :
        leftEnd = right - 1
        k = left
        num = rightEnd - left + 1
        while (left <= leftEnd and right <= rightEnd) :
            if (array[left] <= (array[right])) :
                tmp[k + 1] = array[left + 1]
            else :
                tmp[k + 1] = array[right + 1]
        while (left <= leftEnd) :
            tmp[k + 1] = array[left + 1]
        while (right <= rightEnd) :
            tmp[k + 1] = array[right + 1]
        i = 0
        while (i < num) :
            array[rightEnd] = tmp[rightEnd]
            i += 1
            rightEnd -= 1