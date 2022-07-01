def insertionsort(lista):
    for i in range(1, len(lista)): 
        x=lista[i] 
   
        j=i-1
        while (j>=0 and x<lista[j]): 
                lista[j+1]=lista[j] 
                j-=1
        lista[j+1]=x
