def quicksort_pb(array):
    # init stack, used to avoid recursion
    low, high = 0, len(array)-1
    stack_len = int(numpy.log(high)*20)
    stack_left = [None] * stack_len
    stack_right = [None] * stack_len
    idx_left = 0
    idx_right = -1

    stack_left[0] = (low, high)
    while True:
        while idx_left > -1:
            low, high = stack_left[idx_left]
            idx_left -= 1
            if low < high: break
            if idx_right > -1:
                low, high = stack_right[idx_right]
                idx_right -= 1
                if low < high: break
        else:
            return

        part = int(random() * (high-low) + low)
        counter = low
        #swap(array, part, low)
        array[part], array[low] = array[low], array[part]
        array_low = array[low]
        for i in xrange(low+1, high+1):
            if array[i] < array_low:
                counter += 1
                #swap(array, counter, i)
                array[counter], array[i] = array[i], array[counter]
        #swap(array, low, counter)
        array[low], array[counter] = array[counter], array[low]
        idx_left += 1
        stack_left[idx_left] = (low, counter-1)
        if counter+1 < high:
            idx_right += 1
            stack_right[idx_right] = (counter+1, high)

def quicksort_lc(array):
    if array == []:
        return []
    else:
        pivot = array.pop(randrange(len(array)))
        lesser = quicksort_lc([l for l in array if l < pivot])
        greater = quicksort_lc([l for l in array if l >= pivot])
        return lesser + [pivot] + greater