import numpy as np
import scipy
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math
import scipy.misc
from scipy.io.wavfile import read


a = read("aa.wav")


input1 =np.array(a[1],dtype=float)
input2 = input1


input_feq = np.fft.fft(input1)

n = np.zeros(np.int(len(input1)))
for i in range(np.int(len(input1))):
    n[i] = i

# plt.plot(n,abs(input_feq/800000))
# plt.xlim(0,360)
# plt.show()

print(len(input1))

## pre emphasis
high_pass_output = []
i = 1


high_pass_output.append(np.float(input1[0]))
while i< np.int(len(input1)):
    a = np.float(input1[i]-0.95*input1[i-1])
    high_pass_output.append(a)
    i = i+1


high_pass_output = np.array(high_pass_output)


output_feq = np.fft.fft(high_pass_output)

##### Q1 plotting

# fig, axs = plt.subplots(2)
# fig.suptitle('Pre emphasis of Input', fontsize = 14, fontweight='bold')
# plt.subplots_adjust(hspace=0.2)
# axs[0].set_xlim((0,360))
# axs[0].plot(n, abs(input_feq))
# axs[0].set_title("DFT of Original Input Signal",fontsize = 12, fontweight='bold')
# plt.subplots_adjust(hspace=0.5)
# axs[1].set_xlim((0,360))
# axs[1].plot(n, abs(output_feq))
# axs[1].set_title("DFT of high-pass output Signal (after pre emphasis)", fontsize = 12, fontweight='bold')
# plt.show()

write("pre_emp.wav",8000,high_pass_output)


##########################################################################################################################################

# Question 2


input1 = high_pass_output
slice1 = []
i = 240 
while i < 480:
    slice1.append(input1[i])
    i = i+1

hamm = np.hamming(np.int(len(slice1)))

slice1 = hamm*slice1
Narrow_band_spec= np.fft.fft(slice1)



n = np.zeros(240)
for i in range(240):
    n[i] = i

## Q2 plotting

# plt.title(" Magnitude spectrum of centre slice ",fontsize = 14, fontweight='bold' )

# # Narrow_band_spec
# plt.subplot(1,2,1)
# plt.xlabel("Frequency Samples", fontsize = 12, fontweight='bold')
# plt.ylabel("20*log(|DFT(windowed signal)|)", fontsize = 12, fontweight='bold')
# plt.plot(n,20.0*np.log10(abs(Narrow_band_spec)))
# plt.xlim((0,120))
# plt.show()

##########################################################################################################################################
# Question 3


########### auto correlation 
def autocorr(i):
    ## shifted version
    x_shifted = np.zeros(np.int(len(slice1)))
    k = 0
    x= slice1
    while (i+k) < np.int(len(slice1)):
        x_shifted[k] = x[i+k]
        k =  k+1
    # print(x_shifted)
    x= np.array(x)
    # print(x.shape, x_shifted.shape)
    autocorr_sum  = np.matmul((x.T),x_shifted)

    return autocorr_sum

autocorr_coeffs = np.zeros(11)

for i in range(11):
    autocorr_coeffs[i] = autocorr(i)


Lp_coeff_big_mat = []

a = autocorr_coeffs[0]
reflection_coeff = autocorr_coeffs[1]/a
Pole1 = []
Pole1.append(reflection_coeff)
Lp_coeff_big_mat.append(Pole1)


### calculating higher order coeffs 
E = []
E1 = (1.0-(reflection_coeff)*(reflection_coeff))*a
E.append(a)
E.append(E1)

i = 2
error = []
error.append(autocorr_coeffs[0]-(reflection_coeff * autocorr_coeffs[1] )  )

while i< 11:
    prev_coeff = np.array(Lp_coeff_big_mat[i-2])
    coeff = []
    dum = 0 
    j = 1
    while j < i :
        dum = dum + (prev_coeff[j-1])*(autocorr_coeffs[i-j])
        j = j + 1
    
    k = (autocorr_coeffs[i] - dum)/E[i-1]

    E2  = (1-(k*k))*E[i-1]
    E.append(E2)    
    for m in range(i-1):
        a = prev_coeff[m] - k*prev_coeff[i-m-2]
        coeff.append(a) 
    coeff.append(k)
    Lp_coeff_big_mat.append(coeff)

    ## error calculation
    err = autocorr_coeffs[0]
    for t in range(i):
        err = err - (coeff[t]*autocorr_coeffs[t+1])  
    error.append(err)

    
    i = i+1



n= []

i = 1 
while i < 11:
    n.append(i)
    i = i+1

i = 0
err1 = []
while i<9:
    err1.append(error[i])
    i = i+2


###### Q3 Plotting

# plt.title(" Error signal energy (i.e. square of gain)  vs  p",fontsize = 14, fontweight='bold' )
# # plt.adjust(hspace=5)
# plt.plot(n, error)
# plt.xticks(np.arange(1, 11, 1))
# plt.xlabel("p (no of poles in model)", fontsize = 12, fontweight='bold')
# plt.ylabel("Error Signal energy", fontsize = 12, fontweight='bold')
# plt.show()


#######################################################################################################################################
# Question 4

########## Pole zero Plots
Pole2 = Lp_coeff_big_mat[1]
Pole4 = Lp_coeff_big_mat[3]
Pole6 = Lp_coeff_big_mat[5]
Pole8 = Lp_coeff_big_mat[7]
Pole10 = Lp_coeff_big_mat[9]


G2 = np.sqrt(error[1])
G4 = np.sqrt(error[3])
G6 = np.sqrt(error[5])
G8 = np.sqrt(error[7])
G10 = np.sqrt(error[9])





import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams

# print(Pole10)    
### 6 pole --- Pole zero plot
a= np.zeros(11)
a[0] = G10
b = np.zeros(11)
b[0] = 1
i = 1
while i<11:
    b[i] = -1*Pole10[i-1]
    i = i+1  
print(b)
def zplane(b,a):
    ax = plt.subplot(111)
    uc = patches.Circle((0,0), radius=1, fill=False,color='black', ls='dashed')
    ax.add_patch(uc)
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1
    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
    poly = np.poly1d(a)
    p = poly.r
    z = np.roots(b)
    k = kn/float(kd)  
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g' , label = "Zeros") 
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r', label = "Poles")

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)
    plt.title("Pole- Zero plot for 6 pole model" ,fontsize = 15, fontweight='bold'  )
    plt.legend()
    plt.show()

    

    return  p, k


# zplane(a,b)
# print(zplane(a,b))
### Reference : https://www.dsprelated.com/showcode/244.php


######################################################################################## 
# Question 5
# plt.subplot(1,2,2)
########################## 2 Pole


samp_freq = 8000
n = 1024  
freq_step = (samp_freq/2) / n
freq_response = np.zeros(n)

for n1 in range(n):
    har = n1*freq_step
    z = np.exp(2j*np.pi*har/samp_freq)
    num = G2
    denom = 1 - Pole2[0]/z - Pole2[1]/z**2 
    freq_response[n1]=abs(num/denom)

a = 20*np.log10(freq_response)
b= 20*np.log10(freq_response)
a = a/max(abs(a))
# plt.subplot(1,2,1)
f = np.arange(0,samp_freq/2,freq_step)

##shifted plot
# plt.plot(f, b+40 ,label =  " 2 - Pole ")
#### original plot
# plt.plot(f, b ,label =  "2 Pole ")
# plt.grid(True)


################## 4 -pole system


samp_freq = 8000
n = 1024  
freq_step = (samp_freq/2) / n


freq_response = np.zeros(n)


for n1 in range(n):
    har = n1*freq_step
    z = np.exp(2j*np.pi*har/samp_freq)
    num = G4
    denom = 1 - Pole4[0]/z - Pole4[1]/z**2  - Pole4[2]/(z*z*z)
    freq_response[n1]=abs(num/denom)

a = 20*np.log10(freq_response)
b= 20*np.log10(freq_response)
a = a/max(abs(a))
# plt.subplot(1,2,1)
f = np.arange(0,samp_freq/2,freq_step)


# plt.plot(f, b+30, label = "4 - Pole")
# # plt.plot(f, b, label = "4 Pole")
# plt.grid(True)






############ 6-ploe system


samp_freq = 8000
n = 1024  
freq_step = (samp_freq/2) / n


freq_response = np.zeros(n)


for n1 in range(n):
    har = n1*freq_step
    z = np.exp(2j*np.pi*har/samp_freq)
    num = G6
    denom = 1 - Pole6[0]/z - Pole6[1]/z**2  - Pole6[2]/(z*z*z)  - Pole6[3]/(z*z*z*z)  - Pole6[4]/(z*z*z*z*z) - Pole6[5]/(z*z*z*z*z*z)
    freq_response[n1]=abs(num/denom)

a = 20*np.log10(freq_response)
b= 20*np.log10(freq_response)
a = a/max(abs(a))
# plt.subplot(1,2,1)
f = np.arange(0,samp_freq/2,freq_step)

# 
# plt.plot(f, b+20, label = "6 pole")
# # plt.plot(f, b, label = "6 pole")
# plt.grid(True)





############ 8-ploe system


samp_freq = 8000
n = 1024  
freq_step = (samp_freq/2) / n


freq_response = np.zeros(n)


for n1 in range(n):
    har = n1*freq_step
    z = np.exp(2j*np.pi*har/samp_freq)
    num = G8
    denom = 1 - Pole8[0]/z - Pole8[1]/z**2  - Pole8[2]/(z*z*z)  - Pole8[3]/(z*z*z*z)  - Pole8[4]/(z*z*z*z*z) - (1/(z*z*z*z*z))*(Pole8[5]/z - Pole8[6]/z**2  - Pole8[7]/(z*z*z)  ) 
    freq_response[n1]=abs(num/denom)


a = 20*np.log10(freq_response)
b= 20*np.log10(freq_response)
a = a/max(abs(a))
f = np.arange(0,samp_freq/2,freq_step)


# plt.plot(f, b+10, label = "8 pole")
# # plt.plot(f, b, label = "8 pole")
# plt.grid(True)




############ 10-ploe system


samp_freq = 8000
n = 1024  
freq_step = (samp_freq/2) / n


freq_response = np.zeros(n)


for n1 in range(n):
    har = n1*freq_step
    z = np.exp(2j*np.pi*har/samp_freq)
    num = G10
    denom = 1 - Pole10[0]/z - Pole10[1]/z**2  - Pole10[2]/(z*z*z)  - Pole10[3]/(z*z*z*z)  - Pole10[4]/(z*z*z*z*z) - (1/(z*z*z*z*z))*(Pole10[5]/z - Pole10[6]/z**2  - Pole10[7]/(z*z*z)  - Pole10[8]/(z*z*z*z)  - Pole10[9]/(z*z*z*z*z)) 
    freq_response[n1]=abs(num/denom)

a = 20*np.log10(freq_response)
b= 20*np.log10(freq_response)
a = a/max(abs(a))
# plt.subplot(1,2,2)
f = np.arange(0,samp_freq/2,freq_step)







# plt.plot(f, b, label = "10 pole")
# plt.plot(f, b, label = "10 pole")
# plt.grid(True)
# plt.title("dB magnitude frequency response of the estimated all-pole filter",fontsize = 15, fontweight='bold' )
# plt.xlabel("Frequency (Hz)", fontsize = 12, fontweight='bold')
# plt.ylabel("20*log(|Frequency response|)", fontsize = 12, fontweight='bold')
# plt.legend()
# plt.show()



#######################################################################################################################################

#  Question 6

# Inverse filter

a = np.zeros(11)

Error_out = []

i = 0
le = np.int(len(input1))
while i< le:
    window = np.zeros(10)
    for j in range(10):
        if (i-j-1) > 0:
            window[j] = input1[i-j-1]
        else:
            window[j] = 0
    window = np.array(window)
    dum = np.matmul(window.T , Pole10)
    summ = input1[i] - dum 
    summ = summ/G10
    Error_out.append(summ)
    i = i+1




def error_autocorr(i):
    ## shifted version
    x_shifted = np.zeros(np.int(len(Error_out)))
    k = 0
    x= Error_out
    while (i+k) < np.int(len(Error_out)):
        x_shifted[k] = x[i+k]
        k =  k+1
    # print(x_shifted)
    x= np.array(x)
    # print(x.shape, x_shifted.shape)
    autocorr_sum  = np.matmul((x.T),x_shifted)

    return autocorr_sum

error_autocorr_coeffs = np.zeros(le)

for i in range(le):
    error_autocorr_coeffs[i] = error_autocorr(i)

########################### 
# found that pitch period = 60 samples
# So 7.5 ms ----->  133.3Hz

def orig_autocorr(i):
    ## shifted version
    x_shifted = np.zeros(np.int(len(input2)))
    k = 0
    x= input2
    while (i+k) < np.int(len(input2)):
        x_shifted[k] = x[i+k]
        k =  k+1
    # print(x_shifted)
    x= np.array(x)
    # print(x.shape, x_shifted.shape)
    autocorr_sum  = np.matmul((x.T),x_shifted)

    return autocorr_sum
le1 = np.int(len(input2))
orig_coeff = np.zeros(le1)

for i in range(le1):
    orig_coeff[i] = orig_autocorr(i)

# plt.title("Autocorrelation of Residual Error signal",fontsize = 15, fontweight='bold' )
# plt.xlabel("shift value", fontsize = 12, fontweight='bold')
# plt.ylabel("Acf value", fontsize = 12, fontweight='bold')
# plt.plot(orig_coeff)
# plt.plot(error_autocorr_coeffs)




fig, axs = plt.subplots(2)
fig.suptitle('Acf plots comparision', fontsize = 14, fontweight='bold')
plt.subplots_adjust(hspace=0.2)
axs[0].plot(orig_coeff)
axs[0].set_title("Autocorrelation of Original signal",fontsize = 12, fontweight='bold')

plt.subplots_adjust(hspace=0.5)
axs[1].plot(error_autocorr_coeffs)
axs[1].set_title("Autocorrelation of Residual Error signal", fontsize = 12, fontweight='bold')






plt.show()


###########################################################################################################################
# Question 7

#By comparison It looks like p = 10 is better approximation

### Impulse train of 133.3Hz

le = np.int(len(Error_out))

Impulse = np.zeros(2400)

i = 0
while i<2400:
    Impulse[i]= 1
    i = i+60 


########### difference equation
out_win = np.zeros(10)
output = np.zeros(2400)
coeff  = np.array(Pole10)
for i in range(2400):
    dum = (G10*Impulse[i]) + np.matmul(coeff.T,out_win)
    # dum = Error_out[i] + np.matmul(Pole10,out_win)
    # print(dum)
    output[i] = dum
    j = 0
    for j in range(10):
        if i-j >=0:
            out_win[j] = output[i-j]
        else :
            out_win[j] = 0  


############### de emphasis
input3= output
le = np.int(len(input3))
de_emph = np.zeros(le)
prev_in = 0
for i in range(le):
    # print(prev_in)
    de_emph[i] = input3[i] + 0.95*prev_in
    prev_in = de_emph[i]
    # print(prev_in)


write("synthesized_aa_10.wav",8000,de_emph)




# input2 = input1
# plt.subplot(1,2,1)
# plt.plot(input1)
# plt.xlim(0,400)


# plt.subplot(1,2,2)
# write("synthesized_aa_10.wav",8000,de_emph)
# plt.xlim(0,400)
# plt.plot(de_emph)
# plt.show()



# fig, axs = plt.subplots(2)
# fig.suptitle('LP re-synthesis using 6-pole model', fontsize = 14, fontweight='bold')
# plt.subplots_adjust(hspace=0.2)
# axs[0].set_xlim((0,720))
# axs[0].plot(input2)
# axs[0].set_title("Original Input signal /a/",fontsize = 12, fontweight='bold')
# plt.subplots_adjust(hspace=0.5)
# axs[1].set_xlim((0,720))
# axs[1].plot(de_emph)
# axs[1].set_title("Reconstructed signal /a/", fontsize = 12, fontweight='bold')

# plt.show()