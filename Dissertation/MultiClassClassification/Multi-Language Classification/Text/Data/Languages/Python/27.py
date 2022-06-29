def setSort(self, input) :    
    sq = 1
    while (sq < len(input)) :
        j = 0
        while (j < len(input) - sq) :
            self.setMergeSort(input, j, j + sq - 1, min(j + sq + sq,len(input) - 1))
            j += sq * 2
        sq += sq
        
def setMergeSort(self, input,  start,  mid,  end) :
    tem = [0] * (len(input))
    i = start
    while (i <= end) :
        tem[i] = input[i]
        i += 1
    i = start
    j = mid + 1
    h = start
    while (h <= end) :
        if (i > mid) :
            input[h] = tem[j + 1]
        elif(j > end) :
            input[h] = tem[i + 1]
        elif(tem[i] < tem[j]) :
            input[h] = tem[i + 1]
        elif(tem[i] > tem[j]) :
            input[h] = tem[j + 1]
        h += 1