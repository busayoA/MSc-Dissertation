def  sort( numbers) :
    if (len(numbers) <= 1) :
        return numbers
    middle = int(len(numbers) / 2)
    left_partition = partition(numbers, 0, middle)
    right_partition =  partition(numbers, middle, len(numbers))
    left_partition = sort(left_partition)
    right_partition =  sort(right_partition)
    return merge(left_partition, right_partition)

def  partition( vector,  begIndex,  endIndex) :
    tmp_vector =  []
    i = begIndex
    while (i < endIndex) :
        tmp_vector.append(vector[i])
        i += 1
    return tmp_vector


def  merge( v1,  v2) :
    tmp_vector =  []
    v1c = 0
    v2c = 0
    while (v1c < len(v1) or v2c < len(v2)) :
        if (v1c == len(v1)) :
            tmp_vector.append(v2[v2c])
            v2c += 1
            continue
        elif(v2c == len(v2)) :
            tmp_vector.append(v1[v1c])
            v1c += 1
            continue
        if (v1[v1c] <= v2[v2c]) :
            tmp_vector.append(v1[v1c])
            v1c += 1
        else :
            tmp_vector.append(v2[v2c])
            v2c += 1
    return tmp_vector