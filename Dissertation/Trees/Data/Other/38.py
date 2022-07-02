def bubble_sort(list_of_numbers):
    for i in range(len(list_of_numbers)-1,0,-1):
        for j in range(i):
            if list_of_numbers[j] > list_of_numbers[j+1]:
                placeholder = list_of_numbers[j]
                list_of_numbers[j] = list_of_numbers[j+1]
                list_of_numbers[j+1] = placeholder
            
    return list_of_numbers