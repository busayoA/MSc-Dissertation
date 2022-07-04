def mergeSort(self, n,  s,  e) :
        if (n == None or len(n) == 0) :
            return
        if (s < e) :
            m = int((s + e) / 2)
            self.mergeSort(n, s, m)
            self.mergeSort(n, m + 1, e)
            self.merge(n, s, m, e)

def merge(self, n,  s,  m,  e) :
        n1 = n[s:m+1] 
        n2 = n[m+1:e+1]
        k = s
        i = 0
        j = 0
        while (k <= e and i < len(n1) and j < len(n2)) :
            if (n1[i] < n2[j]) :
                n[k] = n1[i +1]
            else :
                n[k] = n2[j + 1]
            k += 1
        while (i < len(n1)) :
            n[k + 1] = n1[i + 1]
        while (j < len(n2)) :
            n[k + 1] = n2[j + 1]