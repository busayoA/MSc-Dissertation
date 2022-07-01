def partiton(a, left, right):
    i = left + 1 
    pivot = a[left]
    for j in range(left+1, right+1): 
	    if (a[j] < pivot):
		    a[i], a[j] = a[j], a[i] 
		    i = i+1 
    pos = i - 1
    a[left], a[pos] =  a[pos], a[left]
    return pos

def quickSort(a, left, right):
	if(left <  right):
		pivot = partiton(a, left, right)
		quickSort(a, left, pivot-1)
		quickSort(a, pivot+1, right)