def Quicksort(A,p,r):
    if p < r:
        q = Part(A,p,r)
        Quicksort(A,p,q-1)
        Quicksort(A, q + 1, r)        
def Part (A,p,r):
    pivot = A[r] 
    i = p - 1 
    for j in range (p, r-1):
        if A[j] <= pivot: 
            i = i + 1 
            temp = A[j]
            A[j] = A[i] 
            A[i] = temp 
    temp2 = A[r] 
    A[r] = A[i + 1]
    A[i + 1] = temp2 
    return i + 1