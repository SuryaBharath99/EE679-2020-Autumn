import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
from scipy.io.wavfile import write


F1 = 300
B1 = 100

b = [1, 0, 0]
r = math.exp(-1*0.25*0.5*B1*np.pi*0.5*0.001)
Theta = F1*(np.pi)*(1/8000)
cost = math.cos(Theta)

a = [1, -1*r*2*cost, r*r]


################################----->INPUT  SIGNAL TRIANGLE UNIT <-----------#######

input = []
i = 0 
while i<10 :
    input.append(10-i)
    i = i+1

i = 0
while i<114 :
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


i = 0
while i<len(input):
    display.append(input[i])
    ts.append(i+4*len(input))
    i = i+1 


i = 0
while i<len(input):
    display.append(input[i])
    ts.append(i+5*len(input))
    i = i+1 

#################################################


print(len(input))

output = []


# out = 1.00
# out1 = 1+(2*r*cost)
# out2 = 1-(r*r)+2*r*cost + 4*r*r*cost*cost

out = 0
out1 =0
out2 = 0 

n = []
i= 0

freq = 120
no_of_iter = freq/2
k = 0 
while i<60 :
    j = 0
    k = k+1
    while j<len(input):
        out2 = out1
        out1 = out
        out = input[j]+(2*r*cost*out1) -(r*r*out2)
        n.append(((k-1)*len(input))+j)
        output.append(out)
        j = j+1
    
    i = i+1



output = np.array(output)


write("q3_a.wav",16000,output)


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





# plt.plot(n,output)
# # plt.grid(True)
# plt.xlabel("Time Samples")
# plt.title("Complete 0.5sec Output response")
# plt.show()