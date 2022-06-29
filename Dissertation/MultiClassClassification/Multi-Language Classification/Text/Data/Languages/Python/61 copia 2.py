def selection_sort(num_list): # Takes a list with n integer inputself.
    """Apply selection sort algorithm on list input and sorts them."""
    sortedlist = []
    while len(num_list) > 0:
        minvalue = num_list[0]
        for number in num_list:
            if number < minvalue:
                minvalue = number
        print('\nStep ', len(sortedlist) + 1) # Prints steps of sorting algorithm.
        print(num_list)
        print(sortedlist)

        sortedlist.append(minvalue)
        num_list.remove(minvalue)
    print('\nFinish:', numbers)
    print(sortedlist)