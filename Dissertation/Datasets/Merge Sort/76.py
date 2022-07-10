arr = []
def MergeSort(left,  right) :
    if (left == right) :
        return
    mid = int((left + right) / 2)
    MergeSort(left, mid)
    MergeSort(mid + 1, right)
    Merge(left, mid, right)

def Merge(left,  mid,  right) :
    after = [0] * (right - left + 1)
    lb = left
    rb = mid + 1
    count = 0
    while (lb <= mid and rb <= right) :
        if (arr[lb] <= arr[rb]) :
            after[count + 1] = arr[lb + 1]
        else :
            after[count + 1] = arr[rb + 1]
    while (lb <= mid) :
        after[count + 1] = arr[lb + 1]
    while (rb <= right) :
        after[count + 1] = arr[rb + 1]
    i = 0
    while (i < len(after)) :
        arr[left + 1] = after[i]
        i += 1