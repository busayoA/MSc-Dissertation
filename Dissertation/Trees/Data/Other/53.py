def insertion_sort(lst):
    n = len(lst)
    for i in range(n):
        j = i
        while j > 0 and lst[j-1] > lst[j]:
            # swap the current value one space left
            tmp = lst[j]
            lst[j] = lst[j-1]
            lst[j-1] = tmp
            j -= 1
