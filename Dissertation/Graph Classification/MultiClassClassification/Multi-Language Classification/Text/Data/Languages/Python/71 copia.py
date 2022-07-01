def selectionSort(array):
    """
    A function that takes in an array of integers and returns a sorted version of that array using the Selection
    Sort algorithm
    """
    current_index = 0
    while current_index < len(array):
        smallest_index = current_index
        for idx in range(current_index + 1, len(array)):
            if array[idx] < array[smallest_index]:
                smallest_index = idx
        swap(current_index, smallest_index, array)
        current_index += 1
    return array


def swap(i, j, array):
    array[i], array[j] = array[j], array[i]