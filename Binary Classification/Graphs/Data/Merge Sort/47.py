def  sorter(self, array) :
        if (len(array) <= 1) :
            return array
        left = [0] * (int(len(array) / 2))
        right = [0] * (len(array) - len(left))
        sorter(left)
        sorter(right)
        merge(left, right, array)
        return array

def merge( left,  right,  array) :
    firstIndex = 0
    secondIndex = 0
    union = 0
    while (firstIndex < len(left) and secondIndex < len(right)) :
        if (left[firstIndex] <= right[secondIndex]) :
            array[union] = left[firstIndex]
            firstIndex += 1
        else :
            array[union] = right[secondIndex]
            secondIndex += 1
        union += 1
