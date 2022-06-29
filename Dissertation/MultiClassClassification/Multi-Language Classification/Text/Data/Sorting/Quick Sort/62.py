def quick( arr,  low,  up) :
    piv = 0
    temp = 0
    left = 0
    right = 0
    pivot_placed = False
    print("low : " + str(low) + " up :  " + str(up))
    left = low
    right = up
    piv = low
    if (low >= up) :
        return
    print("sub list : ")
    while (pivot_placed == False) :
        while (arr[piv] <= arr[right] and piv != right) :
            right = right - 1
        if (piv == right) :
            pivot_placed = True
        if (arr[piv] > arr[right]) :
            temp = arr[piv]
            arr[piv] = arr[right]
            arr[right] = temp
            piv = right
        while (arr[piv] >= arr[left] and left != piv) :
            left = left + 1
        if (piv == left) :
            pivot_placed = True
        if (arr[piv] < arr[left]) :
            temp = arr[piv]
            arr[piv] = arr[left]
            arr[left] = temp
            piv = left
    print()
    print("pivot placed is  " + str(arr[piv]))
    print()
    quick(arr, low, piv - 1)
    quick(arr, piv + 1, up)