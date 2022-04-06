def insertion_sort(data, draw_data, speed):
    for i in range(1, len(data)):   # traversing through the array
        key = data[i]
        j = i - 1
        while j >= 0 and key < data[j]:
            data[j + 1] = data[j]   # moving the elements
            j -= 1
        data[j + 1] = key