def selectionAlgorithm(numbers):
    """Function that runs the selection algorithm. Feed it with a list of integers and
    it will sort and then return it."""
    vehicle = 0
    temporary = 0
    for num in range(len(numbers)):
        vehicle = num
        for sub in range(num + 1, len(numbers)):
            if numbers[vehicle] > numbers[sub]:
                vehicle = sub
        temporary = numbers[vehicle]
        print("Temporary is ---> ", temporary)
        numbers[vehicle] = numbers[num]
        numbers[num] = temporary
        
    return numbers