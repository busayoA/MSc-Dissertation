def merge( S1,  S2,  S,  comp) :
        i = 0
        j = 0
        while (i + j < len(S)) :
            if (j == len(S2) or (i < len(S1) and comp.compare(S1[i],S2[j]) < 0)) :
                S[i + j] = S1[i+1]
            else :
                S[i + j] = S2[j+1]

def mergeSort( S,  comp) :
        n = len(S)
        if (n < 2) :
            return
        mid = int(n / 2)
        S1 = S[0:mid]
        S2 = S[mid:n]
        mergeSort(S1, comp)
        mergeSort(S2, comp)
        merge(S1, S2, S, comp)