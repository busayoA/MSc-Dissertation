def devide( array) :
    length = len(array)
    if (length == 1) :
        return
    fi = int(length / 2)
    la = length - fi
    array1 = [0] * (fi)
    array2 = [0] * (la)
    i = 0
    while (i < fi) :
        array1[i] = array[i]
        i += 1
    i = 0
    while (i < fi) :
        array2[i] = array[i + fi]
        i += 1
    devide(array1)
    devide(array2)
    conquer(array, array1, array2)

def conquer( array,  array1,  array2) :
    i = 0
    j = 0
    max = len(array)
    while (True) :
        if (i < len(array1) and j < len(array2)) :
            if (array1[i] < array2[j]) :
                array[i + j] = array1[i]
                i += 1
            else :
                array[i + j] = array2[j]
                j += 1
            if (i + j == max) :
                return
        elif(i < len(array1)) :
            array[i + j] = array1[i]
            i += 1
            if (i + j == max) :
                return
        else :
            array[i + j] = array2[j]
            j += 1
            if (i + j == max) :
                return