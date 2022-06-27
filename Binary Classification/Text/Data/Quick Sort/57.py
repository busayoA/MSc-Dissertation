def ordenaArray(iz,  de,  vector) :
        i = 0
        j = 0
        x = 0
        w = 0
        i = iz
        j = de
        x = vector[int((iz + de) / 2)]
        while True :
            while (vector[i] < x) :
                i += 1
            while (x < vector[j]) :
                j -= 1
            if (i <= j) :
                w = vector[i]
                vector[i] = vector[j]
                vector[j] = w
                i += 1
                j -= 1
            if((i <= j) == False) :
                    break
        w = vector[i]
        vector[i] = vector[de]
        vector[de] = w
        if (iz < j) :
            ordenaArray(iz, j, vector)
        if (i < de) :
            ordenaArray(i, de, vector)