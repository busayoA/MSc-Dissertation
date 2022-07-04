def left_to_right(lr_list,pivot_index):
     pointer=0
     while(lr_list[pointer]<lr_list[pivot_index] and pointer!=pivot_index): #loop will exit when left element is Greater than Pivot Element
         pointer+=1                       
     if(pointer==pivot_index):
        #pivot_index is at corrct position

        if(pointer>0):
            lr_list[0:pointer]= right_to_left(lr_list[0:pointer],0)

        lr_list[pointer+1:]=right_to_left(lr_list[pointer+1:],0)
     else:
         #swap(a[t],a[pivot_index])
         lr_list[pointer],lr_list[pivot_index]=lr_list[pivot_index],lr_list[pointer]
         print(lr_list," swaped ",lr_list[pointer]," and ",lr_list[pivot_index])
         lr_list=right_to_left(lr_list,pointer)
     return lr_list

def right_to_left(rl_list,pivot_index):
     t=len(rl_list)-1

     if(len(rl_list)<=1):

         return rl_list

     while(rl_list[t]>rl_list[pivot_index] and t!=pivot_index):
         t-=1

     if(t==pivot_index):
         #pivot_index is at corrct position

        if(t>0): 
            rl_list[0:t]= right_to_left(rl_list[0:t],0)

        rl_list[t+1:]=right_to_left(rl_list[t+1:],0)

     else:
         #swap(a[t],a[pivot_index])
         rl_list[t],rl_list[pivot_index]=rl_list[pivot_index],rl_list[t]
         print(rl_list," swaped ",rl_list[t]," and ",rl_list[pivot_index])
         rl_list=left_to_right(rl_list,t)
     return rl_list