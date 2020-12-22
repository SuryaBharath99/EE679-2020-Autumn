import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
from scipy.io.wavfile import write

b = [1, 0, 0]
r = math.exp(-1*25*np.pi*0.5*0.001)
Theta = 9*(np.pi)*(1/80)
cost = math.cos(Theta)

a = [1, -1*r*2*cost, r*r]






################################----->INPUT  SIGNAL TRIANGLE UNIT <-----------#######

input = []
i = 0 
while i<10 :
    input.append(10-i)
    i = i+1

i = 0
while i<95 :
    input.append(0)
    i = i+1    


i = 0
while i<9:
    input.append(1+i)
    i = i+1


print(len(input))


display = []
ts = []
i = 0
while i<len(input):
    display.append(input[i])
    ts.append(i)
    i = i+1


i = 0
while i<len(input):
    display.append(input[i])
    ts.append(i+len(input))
    i = i+1

i = 0
while i<len(input):
    display.append(input[i])
    ts.append(i+2*len(input))
    i = i+1

i = 0
while i<len(input):
    display.append(input[i])
    ts.append(i+3*len(input))
    i = i+1    

#################################################




output = []

out = 0
out1 =0
out2 = 0 
# N= 10
n = []
i= 0

no_of_iter =114
k = 0 
while i<70 :
    j = 0
    k = k+1
    while j<114:
        out2 = out1
        out1 = out
        out = input[j]+(2*r*cost*out1) -(r*r*out2)
        n.append(((k-1)*114)+j)
        output.append(out)
        j = j+1
    
    i = i+1



output = np.array(output)


write("q2.wav",16000,output)


plt.subplot(1, 2, 1)
plt.plot(ts,display)
plt.ylim((0,15))
plt.xlim((0,400))
plt.xticks(np.arange(0, 400, 20))
plt.xlabel("Time samples")
plt.grid(True)
plt.title("Input periodic Signal")
plt.subplot(1, 2, 2)
plt.plot(n,output)
# plt.ylim((-5,5))
plt.xlim((0,1000))
# plt.xticks(np.arange(0, 250, 20))
# plt.yticks(np.arange(-6, 6, 0.5))
plt.grid(True)
plt.xlabel("Time Samples")
plt.title("Zoomed Output response")
plt.show()



###################-----------> full 0.5 seconds plot <----- ##########
# plt.plot(n,output)
# # plt.grid(True)
# plt.xlabel("Time Samples")
# plt.title("Complete 0.5sec Output response")
# plt.show()