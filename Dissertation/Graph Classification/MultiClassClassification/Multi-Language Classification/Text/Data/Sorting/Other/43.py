def bubble_sort1(A):
    for i in range (0, len(A) - 1):
		for j in range (0, len(A) - i - 1):
			if A[j] > A[j+1]:
				A[j], A[j+1] = A[j+1], A[j]