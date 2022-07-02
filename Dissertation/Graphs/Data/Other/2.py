def bubble_sort(vector):
    for i in range(0, len(vector)):
        for j in range(len(vector) - 1, i, -1):
            if vector[j - 1] > vector[j]:
                vector[j - 1], vector[j] = vector[j], vector[j - 1]
