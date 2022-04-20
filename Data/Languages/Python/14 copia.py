def insertion_sort(integers):
    integers_clone = list(integers)
    for i in range(1, len(integers_clone)):
        j = i
        while integers_clone[j] < integers_clone[j - 1] and j > 0:
            integers_clone[j], integers_clone[j-1] = integers_clone[j-1], integers_clone[j]
            j -= 1
            
    return integers_clone