def swap(A,i1,i2):
      A[i1], A[i2] = A[i2], A[i1]

def partition(A,g,p):
    for j in range(g,p):
      if A[j] <= A[p]:
        swap(A,j,g)
        g += 1
      
    swap(A,p,g)
    return g
    
def _quicksort(A,s,e):
    if s >= e:
      return
    p = partition(A,s,e)
    _quicksort(A,s,p-1) 
    _quicksort(A,p+1,e)

def quicksort(A):
    _quicksort(A,0,len(A)-1)