import random
def quickSort( array,  left,  right,  comp) :

    if (right > left) :
        temp = None
        pivot = array[random.randint(right - left, left)]
        l = left - 1
        r = right
        while True :
            while True : 
                l += 1
                if((comp.compare(array[l],pivot) < 0) == False) :
                    break
            while True : 
                r -= 1
                if((r > l and comp.compare(array[r],pivot) > 0) == False) :
                    break
            temp = array[l]
            array[l] = array[r]
            array[r] = temp
            if((r > l) == False) :
                    break
        array[r] = array[l]
        array[l] = pivot
        array[right] = temp
        quickSort(array, left, l - 1, comp)
        quickSort(array, l + 1, right, comp)