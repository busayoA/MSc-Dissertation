def  sorting( data,  left,  right) :
    if (left < right) :
        center = int((left + right) / 2)
        sorting(data, left, center)
        sorting(data, center + 1, right)
        return merge(data, left, center, right)
    return data

def  merge( data,  left,  center,  right) :
    tmpArr = [0] * (len(data))
    mid = center + 1
    third = left
    tmp = left
    while (left <= center and mid <= right) :
        if (data[left] <= data[mid]) :
            tmpArr[third + 1] = data[left +1]
        else :
            tmpArr[third + 1] = data[mid + 1]
    while (mid <= right) :
        tmpArr[third + 1] = data[mid + 1]
    while (left <= center) :
        tmpArr[third + 1] = data[left + 1]
    while (tmp <= right) :
        data[tmp] = tmpArr[tmp +1]
    return data