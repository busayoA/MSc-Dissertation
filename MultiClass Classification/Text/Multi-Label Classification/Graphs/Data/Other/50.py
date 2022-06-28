def sort(arr):
    arr_edit = arr
    for i in range(1, len(arr)):
        temp = arr_edit[i]

        j = i - 1
        while j >= 0 and temp < arr_edit[j]:
            arr_edit[j + 1] = arr_edit[j]
            j -= 1
        arr_edit[j + 1] = temp
    return arr_edit