#importing libraries
from pylab import *
import scipy.io.wavfile as wavfile  #for reading and writing .wav files
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import sklearn.cluster as sk
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def signaltonoise(a,b):
    c = np.zeros(len(a))
    c = b-a
    P_n = (np.linalg.norm(c))**2
    P_s = (np.linalg.norm(b))**2
    snr = 10*log(P_s/P_n)
    return snr

def lowpass_filter(data,cutoff,fs,order): #data= input, cutoff = filter cutoff freq, fs=sampling freq, order = filter order
    nyq = 0.5*fs #nyquist freq
    normalized_cutoff = cutoff/nyq #done this way since the butterworth filter fn accepts normalized cutoff only
    #filter coefficients
    b, a = sig.butter(order,normalized_cutoff,btype ='low', analog = False) #numerator and denom of filter
    y = sig.lfilter(b,a,data) #returns response from filter
    return y

def Ps(data,p,l_ambda): #predictor based on spectral envelope with p coefficients ,I'm using p<len(data) which is the norm
    r = np.zeros(p)
    a = np.zeros(p) #predictor coefficients
    t = np.zeros(p) #temporary coefficients
    k = np.zeros(p) #reflection coefficients
    x = np.correlate(data,data,'full') #autocorrelation of speech expressed as a vector,has 2*(len)-1 elements
    r = x[len(data)-1:len(data)+p-1]
    
    #having done this, we gotta implement regularization phi_n[i][j] =phi[i][j] +lambda*emin*u[i-j]
    emin = levinson_durbin(r)[1]
    u    = np.zeros(p)    #autocorrelation of high pass filtered white noise
    u[:3]= [3/8,1/4,1/16]
    phi = sp.linalg.toeplitz(r)
    phi_n = np.zeros((p,p))
    
    for i in range(p):
        for j in range(p):
            phi_n[i][j] = phi[i][j] + l_ambda*emin*u[i-j]
    
    L = np.linalg.cholesky(phi_n) #since inversion is time consuming
    r_n = r + l_ambda*emin*u
    
    q = np.linalg.lstsq(L,r_n,rcond = None)[0]
    a = np.linalg.lstsq(L.T,q,rcond = None)[0]
    #print(q)
    
    #generating parcor coefficients or reflection coefficients
    #r_p = (1/np.sqrt((np.abs(data[0])**2 + np.linalg.norm(q)**2)))*q
    
    #a[0] = 1. #initialization
    #e = r_n[0]  # e is the mse 

    #for i in range(1,p):

    #   for j in range(1, i):
    #        k[i-1] = r_p[i-1]
    #        a[i] = k[i-1]

    #    for j in range(len(r)-1):
    #        t[j] = a[j]

    #     for j in range(1, i):
    #        a[j] += k[i-1] * np.conj(t[i-j])

    #    e *= 1 - k[i-1] * np.conj(k[i-1])  #mse

    coeff = np.zeros(p)
    coeff = a  #since its an allpole filter
    return coeff,phi,e

def levinson_durbin(r):   #levinson_durbin recursion to find the predictor coefficients using the lattice method
    # Estimated coefficients
    a = np.zeros(len(r))
    # temporary array
    t = np.zeros(len(r))
    # Reflection coefficients
    k = np.zeros(len(r))

    a[0] = 1. #initialization
    e = r[0]  # e is the mse 

    for i in range(1,len(r)):
        acc = r[i]   #auto correlation coefficient
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(len(r)-1):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])  #mse

    return a, e, k

def Pd(d,order,j,Nupdate,Niter):  #approximate solution to the pitch prediction filter
    #x = np.correlate(d,d,'same') #autocorrelation of speech expressed as a vector,has 2*(len)-1 elements
    #phi= sp.linalg.toeplitz(x)
    #to find m
    Jmin = 10**20 #some high value
    M   = 0
    phi = np.zeros((Nupdate,Nupdate))
    for m in range(int((j-1)*Nupdate)+10,int((j)*Nupdate)):
        i= m - int((j-1)*Nupdate)
        if(j==Niter-1):
            if(m<len(d)):
                phi[0][i] = np.dot(d[int(j*Nupdate):],d[int(j*Nupdate-m):-m])
                phi[i][i] = np.linalg.norm(d[int(j*Nupdate-m):-m])**2
        
                J = (np.linalg.norm(d[int(j*Nupdate):-m])**2) - ((phi[0][i])**2)/phi[i][i]
                if(J<Jmin):
                    Jmin =J
                    M=m
        elif(j==0):
            k = np.zeros(Nupdate)
            return k
        else:        
            phi[0][i] = np.dot(d[int(j*Nupdate):int((j+1)*Nupdate)],d[int(j*Nupdate-m):int((j+1)*Nupdate-m)])
            phi[i][i] = np.linalg.norm(d[int(j*Nupdate-m):int((j+1)*Nupdate-m)])**2
        
            J = (np.linalg.norm(d[int(j*Nupdate):int((j+1)*Nupdate)])**2) - ((phi[0][i])**2)/phi[i][i]
            if(J<Jmin):
                Jmin = J
                M=m
    beta = np.zeros(Nupdate)
    M -= int((j-1)*Nupdate)
    print('M',M)
    for i in range(Nupdate):
        if(i<M):
            beta[i] = 0
        elif(i>(M+order-1)):
            beta[i] = 0
        else:
            beta[i] = -1*phi[0][i]/phi[i][i]
    return beta

def F(alpha,coeff):
    f= np.zeros(len(coeff)+1)
    f[1:]=alpha*coeff
    return f

def Quantizer_out(rms,q_pts,q):#output from the quantizer
    #using lloyd max algorithm through kmeans for optimal quantization
    
    x = np.random.normal(0,rms,1000) #generates gaussian with sigma = rms
    kmeans = sk.KMeans(q_pts,tol = 10**-5).fit(x.reshape(-1,1))  #reshape(-1,1) implemented since kmeans deals only with multi-D
    centroids = np.zeros(q_pts)
    centroids = kmeans.cluster_centers_
    x1 = np.zeros(1)
    x1[0] = q
    index = kmeans.predict(x1.reshape(-1,1))
    ans = centroids[int(index[0])] #q^[n] is given by the centroid closest to q[n]
    
    return ans
    


Audio_in = "TestHello.wav"  #input audio file
fs,data  = wavfile.read(Audio_in)  #fs = initial sampling freq of the source,data = corresponding numpy array
data_l=data[:,0] #left channel
data_r=data[:,1] #right channel
N_sample = len(data_l) #No of samples initially
fs_fin = 7350 #final sampling freq
N = int(N_sample*fs_fin/fs)+1
s_l = np.zeros(N)
s_r = np.zeros(N)

#s = np.zeros((N,2)) #since there are two channels,stereo

#passing it through a second order low pass filter to get rid of higher order frequencies
cutoff = 3400 #3.4 Khz
order = 2
y_l= lowpass_filter(data_l,cutoff,fs,order)
y_r= lowpass_filter(data_r,cutoff,fs,order)

#sampling it at fs_fin
s_l = y_l[::int(fs/fs_fin)]
s_r = y_r[::int(fs/fs_fin)]

s_lowpass = np.zeros((len(s_l),2)) #since there are two channels,stereo

s_lowpass[:,0]=np.int16(s_l)
s_lowpass[:,1]=np.int16(s_r)

wavfile.write("lowpass_out.wav",fs_fin,s_lowpass)

s_l = data_l[::int(fs/fs_fin)]
s_r = data_r[::int(fs/fs_fin)]



#Sending the sampled speech through pre-emphasis filter
emph_filter = [1,-0.4] #pre-emphasis filter numerator
#outputs
s1_l = sig.lfilter(emph_filter,1,s_l) 
s1_r = sig.lfilter(emph_filter,1,s_r)
#s1_l = s_l
#s1_r = s_r

Nsam = len(s1_l) #no of samples totally
time = Nsam/fs_fin
t_param = 10**-2  #the period at which we routinely update the filter parameters
Nupdate = int(fs_fin*t_param) #no of samples before an update
j=0 #flag which tells how many times filters have been updated
Niter = int(time/t_param)+1  #no of times we've to update the filters
#Npiece= int(fs_fin*t_param)       #no of samples dealt at a time
#print(Npiece)
q_pts = 3                   #quantizer levels
s1_lh = np.zeros(len(s1_l)) #the estimates
s1_rh = np.zeros(len(s1_l))

#initialising the set of signals used in the iterations

#dl     = np.zeros(Nsam) #dn
d_l    = np.zeros(Nsam) #d'n
#dr     = np.zeros(Nsam) 
d_r    = np.zeros(Nsam)
b_l    = np.zeros(Nsam) #intermediate signal, b[n] = q^[n]+d'[n-1]
b_r    = np.zeros(Nsam)
ql     = np.zeros(Nsam) #qn
qr     = np.zeros(Nsam)
q_l    = np.zeros(Nsam) #q^n
q_r    = np.zeros(Nsam) 
dest_l = np.zeros(Nsam) #d^n
dest_r = np.zeros(Nsam)
fl     = np.zeros(Nsam) #fn
fr     = np.zeros(Nsam)
delta_l= np.zeros(Nsam) #intermediate signal, delta[n] = q^[n]-q[n]
delta_r= np.zeros(Nsam)
Ps_order = 10
#Ps_l   = np.zeros((Niter,Ps_order))
#Ps_r   = np.zeros((Niter,Ps_order))
#Pd_l   = np.zeros((Niter,Nupdate))
#Pd_r   = np.zeros((Niter,Nupdate))
#F_l    = np.zeros((Niter,(Ps_order+1)))
#F_r    = np.zeros((Niter,(Ps_order+1)))
spred_r= np.zeros(Nsam) #predicted through Ps
spred_l= np.zeros(Nsam)
for n in tqdm(range(Nsam)):
    if(n%Nupdate == 0): #update all the filters when n is a multiple of Nupdate
        sl     = np.zeros(Nupdate) #s[n]
        sr     = np.zeros(Nupdate)
        dl1     = np.zeros(Nupdate) #d[n]
        dr1     = np.zeros(Nupdate)
        spredl1 = np.zeros(Nupdate)
        spredr1 = np.zeros(Nupdate)
        #last initialization
        if(j==Niter-1):
            Nex = Nsam-j*Nupdate #All the leftover values to prevent size mismatch in evaluation
            sl[0:Nex] = s1_l[j*Nupdate:]
            sr[0:Nex] = s1_r[j*Nupdate:]
            
        else:
            sl     = s1_l[j*Nupdate:(j+1)*Nupdate]
            sr     = s1_r[j*Nupdate:(j+1)*Nupdate]
        
        #updating rms
        if(n%(2*Nupdate) == 0):
            if(j):
                rms_l = np.sqrt(mean_squared_error(s1_l[(j-1)*Nupdate:(j)*Nupdate],s1_lh[(j-1)*Nupdate:(j)*Nupdate]))
                rms_r = np.sqrt(mean_squared_error(s1_r[(j-1)*Nupdate:(j)*Nupdate],s1_rh[(j-1)*Nupdate:(j)*Nupdate]))
                print(rms_l,rms_r)
            
            
        #getting the filter coefficients of Ps
        Ps_order = 10
        Ps_lambda = 0.05
        Ps_l[j],phi_l,e_l = Ps(sl,Ps_order,Ps_lambda)
        Ps_r[j],phi_r,e_r = Ps(sr,Ps_order,Ps_lambda)
        #print(len(Ps_l[j]))
        #updating d
        spredl1 = sig.lfilter(Ps_l[j],1,sl)
        spredr1 = sig.lfilter(Ps_r[j],1,sr)
        dl1 =sl - spredl1
        dr1 =sr - spredr1
    
        #last initialization,update d
        if(j==Niter-1):
            dl[j*Nupdate:]=dl1[0:Nex]
            dr[j*Nupdate:]=dr1[0:Nex]
            spred_l[j*Nupdate:]=spredl1[0:Nex]
            spred_r[j*Nupdate:]=spredr1[0:Nex]
    
        else:
            dl[j*Nupdate:(j+1)*Nupdate]=dl1
            dr[j*Nupdate:(j+1)*Nupdate]=dr1
            spred_l[j*Nupdate:(j+1)*Nupdate]=spredl1
            spred_r[j*Nupdate:(j+1)*Nupdate]=spredr1
            
        #getting the filter coefficients of Pd
        Pd_order = 3
        Pd_l[j] = Pd(dl,Pd_order,j,Nupdate,Niter)
        Pd_r[j] = Pd(dr,Pd_order,j,Nupdate,Niter)
        #print(len(Pd_r[j]))
        
        #getting the filter coefficients of F
        alpha = 0.5
        F_l[j] = F(alpha,Ps_l[j])
        F_r[j] = F(alpha,Ps_r[j])
        
        #Update j flag
        j+=1
        
        #end of the updating filters loop 
    
    #now to continue with routine signal updates (the recursive filter part)
    
    #Encoder side
    print(j)
    #updating q[n]
    if(n):
        ql[n]   = dl[n] - d_l[n-1] - fl[n-1]
        qr[n]   = dr[n] - d_r[n-1] - fr[n-1]
    else:
        ql[n]   = dl[n] 
        qr[n]   = dr[n] 
    
    #updating q^[n]
    if(j>1): #anything but the first set of samples
            q_l[n] = Quantizer_out(rms_l,q_pts,ql[n])
            q_r[n] = Quantizer_out(rms_r,q_pts,qr[n])
    else:
            q_l[n] = ql[n]
            q_r[n] = qr[n]
            
    #updating delta[n] = q^[n] - q[n]
    delta_l[n] = q_l[n] - ql[n]
    delta_r[n] = q_r[n] - qr[n]
    
    #updating b[n] = q^[n] +d'[n-1]
    if(n):
        b_l[n] = q_l[n] + d_l[n-1]
        b_r[n] = q_r[n] + d_r[n-1]
    else:
        b_l[n] = q_l[n] 
        b_r[n] = q_r[n]
        
    #updating f[n]
    #convolving delta[n] explicitly with the filter
    if(j==1):
        if(n):
            for k in range(len(F_l[j-1])):
                if((n-k)>0):
                    fl[n] += delta_l[n-k]*F_l[j-1][k]
                    fr[n] += delta_r[n-k]*F_r[j-1][k]
    else:
        for k in range(len(F_l[j-1])):
            fl[n] += delta_l[n-k]*F_l[j-1][k]
            fr[n] += delta_r[n-k]*F_r[j-1][k]
            
        if(fr[n]>2*rms_r):
            fr[n] = 2*rms_r
        if(fl[n]>2*rms_l):
            fl[n] = 2*rms_l
    
    #updating d'[n]
    #convolving b[n] explicitly with the filter Pd
    if(j==1):
        if(n):
            for k in range(Nupdate):
                if((n-k)>0):
                    d_l[n] += b_l[n-k]*Pd_l[j-1][k]
                    d_r[n] += b_r[n-k]*Pd_r[j-1][k]
    else:
        for k in range(Nupdate):
            d_l[n] += b_l[n-k]*Pd_l[j-1][k]
            d_r[n] += b_r[n-k]*Pd_r[j-1][k]
    
    
    #Decoder side
    
    #updating d^[n]
    #convolving d^[n-1] explicitly with the filter Pd
    if(j==1):
        if(n):
            for k in range(Nupdate):
                if((n-k)>0):
                    dest_l[n] += dest_l[n-k]*Pd_l[j-1][k]
                    dest_r[n] += dest_r[n-k]*Pd_r[j-1][k]
            dest_l[n] += q_l[n]
            dest_r[n] += q_r[n]
    else:
        for k in range(Nupdate):
            if((n-k)>0):
                dest_l[n] += dest_l[n-k]*Pd_l[j-1][k]
                dest_r[n] += dest_r[n-k]*Pd_r[j-1][k]
        dest_l[n] += q_l[n]
        dest_r[n] += q_r[n]
        
    #updating s^[n]
    #convolving s^[n-1] explicitly with filter Ps
    if(j==1):
        if(n):
            for k in range(len(Ps_l[j-1])):
                if((n-k)>0):
                    s1_lh[n] += s1_lh[n-k]*Ps_l[j-1][k]
                    s1_rh[n] += s1_rh[n-k]*Ps_r[j-1][k]
            s1_lh[n] += dest_l[n]
            s1_rh[n] += dest_r[n]
    else:
        for k in range(len(Ps_l[j-1])):
            if((n-k)>0):
                s1_lh[n] += s1_lh[n-k]*Ps_l[j-1][k]
                s1_rh[n] += s1_rh[n-k]*Ps_r[j-1][k]
        s1_lh[n] += dest_l[n]
        s1_rh[n] += dest_r[n]


        
#exiting for loop

#sending the estimated speech signal through the de-emphasis filter
s1_lest = sig.filtfilt(1,emph_filter,s1_lh) 
s1_rest = sig.filtfilt(1,emph_filter,s1_rh)

s_est = np.zeros((len(s1_lest),2)) #since there are two channels,stereo

s_est[:,0]=np.int16(s1_lest)
s_est[:,1]=np.int16(s1_rest)

wavfile.write("speech_out.wav",fs_fin,s_est)

"""
fig2 = plt.figure()
fig2.add_subplot(2,1,1) #rows columns index
w3,h3 = sp.signal.freqz(s1_lest1) #emph left #freq response where w is freq and h is the response
plt.plot(w3,10*log10(h3),label='Left Channel')
plt.grid()
plt.legend()
plt.title('Predicted Speech Spectrum including F')
plt.xlabel('$\omega$ (rad/sample)')
plt.ylabel('dB')

fig2.add_subplot(2,1,2) #rows columns index
w4,h4 = sp.signal.freqz(s1_lest2) #emph left #freq response where w is freq and h is the response
plt.plot(w4,10*log10(h4),label='Right Channel')
plt.legend()
plt.grid()
plt.xlabel('$\omega$ (rad/sample)')
plt.ylabel('dB')

fig2.show()

snr_l = signaltonoise(s1_lest1,s1_l)
snr_r = signaltonoise(s1_lest2,s1_r)
print(snr_l,snr_r)

fig2 = plt.figure()
fig2.add_subplot(2,1,1) #rows columns index
w3,h3 = sp.signal.freqz(s1_rest1) #emph left #freq response where w is freq and h is the response
plt.plot(w3,10*log10(h3),label='Left Channel')
plt.grid()
plt.legend()
plt.title('Predicted Speech Spectrum with F=Ps')
plt.xlabel('$\omega$ (rad/sample)')
plt.ylabel('dB')

fig2.add_subplot(2,1,2) #rows columns index
w4,h4 = sp.signal.freqz(s1_rest2) #emph left #freq response where w is freq and h is the response
plt.plot(w4,10*log10(h4),label='Right Channel')
plt.legend()
plt.grid()
plt.xlabel('$\omega$ (rad/sample)')
plt.ylabel('dB')

fig2.show()

snr_l = signaltonoise(s1_rest1,s1_l)
snr_r = signaltonoise(s1_rest2,s1_r)
print(snr_l,snr_r)
"""