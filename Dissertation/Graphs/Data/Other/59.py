from random import randint

def insertionSort(li):
	for i in range (1, len(li)):
		# Look for out-of-order element:
		if li[i] < li[i-1]:
			# Swap that element until it reaches the right place:
			for j in range (i, 0, -1):
				if li[j] < li[j-1]:
					temp = li[j]
					li[j] = li[j-1]
					li[j-1] = temp