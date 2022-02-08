# Source: https://gist.github.com/femmerling/4364026

def quicksort(input_list):
    higher = []
    lower = []
    if len(input_list) > 2:	
	    pivot = (len(input_list) - 1)/2	
	    mid = [input_list[pivot]]
	    i = 0
	    while i < len(input_list):
            if i != pivot:
                if input_list[i] <= input_list[pivot]:
                    lower.append(input_list[i])
                elif input_list[i] > input_list[pivot]:
                    higher.append(input_list[i])
	    i=i+1
	    return quicksort(lower)+mid+quicksort(higher)
    elif len(input_list) == 2:
        if input_list[0] > input_list[1]:
            input_list[0],input_list[1] = input_list[1],input_list[0]
            return input_list
        else:
	        return input_list