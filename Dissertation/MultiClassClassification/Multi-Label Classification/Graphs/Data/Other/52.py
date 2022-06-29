my_arr = [3,5,1,7,2,9,8,4,6]

for i in range(len(my_arr)): 
    min_idx = i 
    for j in range(i+1, len(my_arr)): 
        if my_arr[min_idx] > my_arr[j]: 
            min_idx = j
    temp = my_arr[i]
    my_arr[i] = my_arr[min_idx]
    my_arr[min_idx] = temp
