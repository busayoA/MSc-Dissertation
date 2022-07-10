def sort(values) :
    numbers = values
    number = len(values)
    helper = [0] * (number)
    mergesort(0, number - 1, helper, numbers)

def mergesort(low,  high, helper, numbers) :
    if (low < high) :
        middle = low + int((high - low) / 2)
        mergesort(low, middle)
        mergesort(middle + 1, high)
        i = low
        while (i <= high) :
            helper[i] = numbers[i]
            i += 1
        i = low
        j = middle + 1
        k = low
        while (i <= middle and j <= high) :
            if (helper[i] <= helper[j]) :
                numbers[k] = helper[i]
                i += 1
            else :
                numbers[k] = helper[j]
                j += 1
            k += 1
        while (i <= middle) :
            numbers[k] = helper[i]
            k += 1
            i += 1