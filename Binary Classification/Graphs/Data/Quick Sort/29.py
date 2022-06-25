def sort(self, values) :
    if (values == None or len(values) == 0) :
        return
    self.numbers = values
    self.number = len(values)
    self.quicksort(0, self.number - 1)

def quicksort(self, low,  high) :
    i = low
    j = high
    pivot = self.numbers[low + int((high - low) / 2)]
    while (i <= j) :
        while (self.numbers[i] < pivot) :
            i += 1
        while (self.numbers[j] > pivot) :
            j -= 1
        if (i <= j) :
            self.exchange(i, j)
            i += 1
            j -= 1
    if (low < j) :
        self.quicksort(low, j)
    if (i < high) :
        self.quicksort(i, high)
        
def exchange(self, i,  j) :
    temp = self.numbers[i]
    self.numbers[i] = self.numbers[j]
    self.numbers[j] = temp