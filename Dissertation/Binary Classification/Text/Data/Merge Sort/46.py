def Sort( array,  num) :
    workArray = [0] * (len(array))
    Sort(array, workArray, 0, num)

def sort( array,  workArray,  start,  count) :
    if (count < 2) :
        return
    Sort(array, workArray, start, int(count / 2))
    Sort(array, workArray, start + int(count / 2), count - int(count / 2))
    Merge(array, workArray, start, int(count / 2), start + int(count / 2), count - int(count / 2))

def Merge( array,  workArray,  leftStart,  leftCount,  rightStart,  rightCount) :
    i = leftStart
    j = rightStart
    leftBound = leftStart + leftCount
    rightBound = rightStart + rightCount
    index = leftStart
    while (i < leftBound or j < rightBound) :
        if (i < leftBound and j < rightBound) :
            if (array[j] < array[i]) :
                workArray[index] = array[j + 1]
            else :
                workArray[index] = array[i + 1]
        elif(i < leftBound) :
            workArray[index] = array[i + 1]
        else :
            workArray[index] = array[j + 1]
        index += 1
    i = leftStart
    while (i < index) :
        array[i] = workArray[i]
        i += 1