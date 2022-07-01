def sort( data,  l,  r) :
    left = l
    right = r
    pivot = data[int((l + r) / 2)]
    while True :
        while (data[left] < pivot) :
            left += 1
        while (data[right] > pivot) :
            right -= 1
        if (left <= right) :
            temp = data[left]
            data[left] = data[right]
            data[right] = temp
            left += 1
            right -= 1
        if((left <= right) == False) :
                break
    if (l < right) :
        sort(data, l, right)
    if (r > left) :
        sort(data, left, r)