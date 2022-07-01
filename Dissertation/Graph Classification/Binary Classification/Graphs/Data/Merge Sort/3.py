def merge_sort(array):
    if (len(array) == 1 or len(array) == 0):
        return array
    left_part, right_part = array[ : len(array) // 2], array[len(array) // 2 : ]
    merge_sort(left_part)
    merge_sort(right_part) 
    
    sorted_array = merge(left_part, right_part)
    for i in range(len(array)):
        array[i] = sorted_array[i]

def merge(left_part, right_part):
    i = 0
    j = 0
    k = 0
    sorted_array = [0] * (len(left_part) + len(right_part))
    
    while (i < len(left_part) and j < len(right_part)):
        if (left_part[i] <= right_part[j]):
            sorted_array[k] = left_part[i]
            i += 1
        else:
            sorted_array[k] = right_part[j]
            j += 1
        k += 1
        
    while (i < len(left_part)):
        sorted_array[k] = left_part[i]
        i += 1
        k += 1
        
    while (j < len(right_part)):
        sorted_array[k] = right_part[j]
        j += 1
        k += 1
    
    return sorted_array