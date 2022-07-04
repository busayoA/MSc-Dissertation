def sort(listo):
    for i in range(len(listo)):
        num=listo[i]
        listo.pop(i)
        whiind=0
        while listo[whiind]<num and whiind<i:
            whiind+=1
        listo.insert(whiind,num)
    print(listo)
 