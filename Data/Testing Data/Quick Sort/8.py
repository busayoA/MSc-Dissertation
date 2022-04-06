def quickSort(A, si, ei):
    if si < ei:
        pi=partition(A,si,ei)
        quickSort(A,si,pi-1)
        quickSort(A,pi+1,ei)


def partition(A, si, ei):
    x = A[ei]
    i = (si-1)
    for j in range(si,ei):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
 
    A[i+1], A[ei] = A[ei], A[i+1]
         
    return i+1
