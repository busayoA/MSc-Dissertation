def selection(L):
                for i in range(len(L)):
                                bigIndex=0
                                for j in range(len(L)-i):
                                                if L[bigIndex]<L[j]:
                                                                bigIndex=j
                                L[bigIndex],L[j]=L[j],L[bigIndex]
                print(L)
