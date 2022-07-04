def insertionSort(arr):
    count=0
    for i in range(1,len(arr)):
        if arr[i] < arr[i-1]: #A lot of the given inputs had the next element as the local max
            a=arr.pop(i)
            m = binaryInsert(arr, a, i-1, 0)
            count += i-m
    return count

def binaryInsert(arr, a, R, L=0):
    while R-L >1:
        m = (L+R)//2
        if a < arr[m]:
            R = m-1
        elif a> arr[m]:
            L = m +1
        else:
            L = m
            break
    index=R+1
    for i in range(L, R+1):
        if a < arr[i]:
            index=i
            break 
    arr.insert(index,a)
    return index