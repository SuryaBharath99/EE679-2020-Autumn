import matplotlib.pyplot as plt
import math
import numpy as np

b = [1, 0, 0]
r = math.exp(-1*25*np.pi*0.5*0.001)
Theta = 9*(np.pi)*(1/80)
cost = math.cos(Theta)


print(-1*r*2*cost)
print(r*r)
a = [1, -1*r*2*cost, r*r]

samp_freq = 16000
n = 1024  
freq_step = (samp_freq/2) / n


freq_response = np.zeros(n)


for n1 in range(n):
    har = n1*freq_step
    z = np.exp(2j*np.pi*har/samp_freq)
    d = a[0] + a[1]/z + a[2]/z**2
    n = b[0] + b[1]/z + b[2]/z**2
    freq_response[n1]=abs(n/d)


f = np.arange(0,samp_freq/2,freq_step)
plt.plot(f, 20*np.log10(freq_response))
plt.ylim((-15,35))
plt.xlim((0,9000))
plt.xticks(np.arange(0, 9000, 100))
plt.xticks(np.arange(-15, 35, 5))
plt.grid(True)
# plt.show()



impulse_response = []

imp = 1.00
out = 1.00
out1 = 1+(2*r*cost)
out2 = 1-(r*r)+2*r*cost + 4*r*r*cost*cost

N= 400
n = []
i= 3
print(r,cost)
impulse_response.append(out)
impulse_response.append(out1)
impulse_response.append(out2)
n.append(0)
n.append(1)
n.append(2)
while i<N :
    
    out = (2*r*cost*out1) -(r*r*out2)
    out2 = out1
    out1 = out
    n.append(i)
    impulse_response.append(out)
    
    i = i+1

# print(impulse_response)
plt.plot(n,impulse_response)



plt.subplot(1, 2, 1)
plt.plot(f, 20*np.log10(freq_response))
plt.ylim((-15,35))
plt.xlim((0,9000))
plt.xticks(np.arange(0, 9000, 500))
plt.yticks(np.arange(-15, 35, 2.5))
plt.xlabel("Frequency")
plt.grid(True)
plt.title("Magnitude of Freq response")
plt.subplot(1, 2, 2)
plt.plot(n,impulse_response)
plt.ylim((-5,5))
plt.xlim((0,300))
plt.xticks(np.arange(0, 250, 20))
plt.yticks(np.arange(-6, 6, 0.5))
plt.grid(True)
plt.xlabel("Time Samples")
plt.title("Impulse response")
plt.show()