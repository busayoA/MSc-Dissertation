def partition_of_quickSort(List, begin, end):
    temp = begin #pivot
    for i in range(begin+1, end+1):
        if List[i] <= List[begin]:
            temp += 1
            List[i], List[temp] = List[temp], List[i]
    List[temp], List[begin] = List[begin], List[temp]
    return temp

def quicksort(List, begin=0, end=None):
    if end is None:
        end = len(List) - 1
    if begin >= end:
        return
    temp = partition_of_quickSort(List, begin, end)
    quicksort(List, begin, temp-1)
    quicksort(List, temp+1, end)

def temp( A, x, y ):
  tmp = A[x]
  A[x] = A[y]
  A[y] = tmp
