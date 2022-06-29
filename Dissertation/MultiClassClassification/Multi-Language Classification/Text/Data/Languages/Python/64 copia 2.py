def selectionsort(list):
    	#print(list)
	for num in range (len(list)-1,0,-1):
		pmax = 0
		for i in range(1, num+1):
			#print "print i", i, "printing pmax", pmax
			if list[i] > list[pmax]:
				pmax = i		
		temp = list[num]
		list[num] = list[pmax]
		list[pmax] = temp
	print(list)	