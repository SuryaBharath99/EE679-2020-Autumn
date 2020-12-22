import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
from scipy.io.wavfile import write


freq = 120
F1 = 730
F2 = 1090
F3 = 2440

b = [1, 0, 0]
r = math.exp(-1*25*0.5*np.pi*0.5*0.001)
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
k = (16000/freq)-19
while i<k :
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






plt.subplot(1, 4, 1)
plt.plot(ts,display)
plt.ylim((0,15))
plt.xlim((0,400))
plt.xticks(np.arange(0, 400, 50))
plt.xlabel("Time samples")
plt.grid(True)
plt.title("Input periodic Signal")









#################################################




output = []
input = np.array(input)
input = input
out = 0
out1 =0
out2 = 0 

n = []
i=0
k = 0 

no_of_iter = freq/2
while i<no_of_iter :
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

plt.subplot(1, 4, 2)
plt.plot(n,output)
plt.grid(True)
plt.xlim((0,1000))
plt.xlabel("Time Samples")
plt.title("Filter1 Response")

print(output)

############################################################################################################
n = []
output1 = []
r = math.exp(-1*25*0.5*np.pi*0.5*0.001)
Theta = F2*(np.pi)*(1/8000)
cost = math.cos(Theta)
l = len(output)
print(len(output))

i = 0
out  = 0
out1 = 0
out2 = 0




while i < l :
    out2 = out1
    out1 = out
    out = output[i]+(2*r*cost*out1) -(r*r*out2)
    n.append(i)
    output1.append(out)
    i = i+1


plt.subplot(1, 4, 3)
plt.plot(n,output1)
plt.grid(True)
plt.xlim((0,1000))
plt.xlabel("Time Samples")
plt.title("Filter2 response")

###############################################################

r = math.exp(-1*25*0.5*np.pi*0.5*0.001)
Theta = F3*(np.pi)*(1/8000)
cost = math.cos(Theta)
l = len(output1)
print(len(output1))
output2 = []
i = 0
out = 0
out1 = 0
out2 = 0

n = []


while i < l :
    out2 = out1
    out1 = out
    out = output1[i]+(2*r*cost*out1) -(r*r*out2)
    n.append(i)
    output2.append(out)
    i = i+1




plt.subplot(1, 4, 4)
plt.plot(n,output2)
plt.grid(True)
plt.xlim((0,1000))
plt.xlabel("Time Samples")
plt.title("Out response")


output = np.array(output2)


write("q4_a_120Hz.wav",16000,output)    

plt.show()