def selectionSort(list):
    n = len(list)
    for i in range(n):
        minimumIndex = i
        for j in range(i + 1, n):
            if list[j] < list[minimumIndex]:
                minimumIndex = j
        list[i], list[minimumIndex] = list[minimumIndex], list[i]
    return list