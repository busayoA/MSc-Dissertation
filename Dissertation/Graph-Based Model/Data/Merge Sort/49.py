def merge( A,  p,  q,  r) :
        n1 = q - p + 1
        n2 = r - q
        B = [0] * (n1)
        C = [0] * (n2)
        i = 0
        while (i < n1) :
            B[i] = A[p + i]
            i += 1
        i = 0
        while (i < n2) :
            C[i] = A[q + 1 + i]
            i += 1
        a = p
        b = 0
        c = 0
        while (b < n1 and c < n2) :
            if (B[b] <= C[c]) :
                A[a] = B[b]
                b += 1
            else :
                A[a] = C[c]
                c += 1
            a += 1
        while (b < n1) :
            A[a] = B[b]
            b += 1
            a += 1
        while (c < n2) :
            A[a] = C[c]
            c += 1
            a += 1

def mergeSort( A,  p,  r) :
    if (p < r) :
        q = int((p + r) / 2)
        mergeSort(A, p, q)
        mergeSort(A, q + 1, r)
        merge(A, p, q, r)