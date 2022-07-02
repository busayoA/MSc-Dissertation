import random
def  quicksort( a,  i,  j) :
    if (j > i) :
        p_rand = int((random() * (j - i) + i))
        p = partition(a, i, j, p_rand)
        quicksort(a, i, p - 1)
        quicksort(a, p + 1, j)
    return int(a)


def  partition( a,  start,  end,  pivot) :
    value = a[pivot]
    swap(a, pivot, end)
    val = start
    i = start
    while (i < end) :
        if ((a[i]) <= (value)) :
            swap(a, i, val)
            val += 1
        i += 1
    swap(a, end, val)
    return val


def  toString( a) :
    t = "["
    i = 0
    while (i < len(a)) :
        t += " " + str(a[i])
        i += 1
    t += "]"
    return t


def  generateArray( i) :
    array = [0] * (i)
    j = 0
    while (j < len(array)) :
        array[j] = random.randint
        j += 1
    return array


def quicksort( a) :
    a = quicksort(a, 0, len(a) - 1)


def swap( a,  i,  j) :
    temp = a[i]
    a[i] = a[j]
    a[j] = temp