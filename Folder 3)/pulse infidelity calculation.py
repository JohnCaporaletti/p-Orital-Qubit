import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from qutip import *
import pandas as pd
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip_qip.algorithms import qft
import qutip_qtrl.logging_utils as logging
logger =logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level=logging.INFO
#QuTiP control modules
import qutip_qtrl.pulseoptim as cpo
import qutip_qtrl.pulsegen as pulsegen

# Constants
eps0=8.8541878128000000 * 10**(-12)
epsr=11.6
e=1.602176634000000 * 10**(-19)
h=6.62607015000000 * 10**(-34)
hb=6.62607015000000 * 10**(-34)/(2*np.pi)
m=.19*9.1093837015 * 10**(-31)
def orb(l):
    return 10**18*(h/(2*np.pi))**2/(m*l**2)

# Gates
X=sigmax()
Y=sigmay()
Z=sigmaz()
I=identity(2)
S=Qobj([[1,0],[0,np.exp(1j*(np.pi/2))]])
H=(1/np.sqrt(2))*Qobj([[1,1],[1,-1]])
T=Qobj([[1,0],[0,np.exp(1j*(np.pi/4))]])
CNOT=Qobj([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
iSWAP=Qobj([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,1,0]])
BSWAP=Qobj([[0,0,0,1j],[0,1,0,0],[0,0,1,0],[1j,0,0,0]])

SI=tensor(S,I)
IS=tensor(I,S)
TI=tensor(T,I)
IT=tensor(I,T)
HI=tensor(H,I)
IH=tensor(I,H)
XI=tensor(X,I)
IX=tensor(I,X)
YI=tensor(Y,I)
IY=tensor(I,Y)
ZI=tensor(Z,I)
IZ=tensor(I,Z)
XX=tensor(X,X)
YY=tensor(Y,Y)
ZZ=tensor(Z,Z)
II=tensor(I,I)
XY=tensor(X,Y)
YX=tensor(Y,X)
XZ=tensor(X,Z)
ZX=tensor(Z,X)
YZ=tensor(Y,Z)
ZY=tensor(Z,Y)
CNOT1=(1j * (np.pi/4) * tensor(I-Z,I-X)).expm()
BSWAP1=(1j * (np.pi/4) * (XX-YY)).expm()

###############################################################################################

# read in pulse data (directory pathway should be changed to reflect the your pulse data location)
BSWAP_pulse=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\NoB\bSWAP.txt",sep='\t', header=None)
HI_pulse=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\NoB\HI.txt",sep='\t', header=None)
TI_pulse=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\NoB\TI.txt",sep='\t', header=None)
SI_pulse=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\NoB\SI.txt",sep='\t', header=None)
II_pulse=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\NoB\II.txt",sep='\t', header=None)

BSWAP_pulse_B=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\B\bSWAP.txt",sep='\t', header=None)
HI_pulse_B=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\B\HI.txt",sep='\t', header=None)
TI_pulse_B=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\B\TI.txt",sep='\t', header=None)
SI_pulse_B=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\B\SI.txt",sep='\t', header=None)
II_pulse_B=pd.read_csv(r"C:\Users\KestnerPC3\GRAPE Pulses\Low Pass Filter Pulses\B\II.txt",sep='\t', header=None)

#Create list of pulse data and target gates
pulse_data_list=[BSWAP_pulse_B,HI_pulse_B,TI_pulse_B,SI_pulse_B,II_pulse_B,BSWAP_pulse,HI_pulse,TI_pulse,SI_pulse,II_pulse]
gate_list=[BSWAP1,HI,TI,SI,II,BSWAP1,HI,TI,SI,II]

#function definitions
#Hamiltonian
def H(fZ,cXX,cYY,cXI,cIX,cYI,cIY):
    return np.pi * (fZ*(IZ+ZI) + cXX*XX + cYY*YY + cXI*XI + cIX*IX + cYI*YI + cIY*IY)

#Compute final unitary
def final_U(fZ,cXX,cYY,delXI,delIX,delYI,delIY,delXX,delYY,pulse_data):
    for i in range(0, len(pulse_data)):
        if i == 0:
            U=II
            t0=pulse_data[0][1]
            #t0 is duration of time step
        if len(pulse_data.columns) == 3:
            U=(-1j * H(fZ,cXX+delXX,cYY+delYY,delXI+pulse_data.iat[i,1],delIX+pulse_data.iat[i,2],delYI,delIY) * t0).expm()*U
        if len(pulse_data.columns) == 5:
            U=(-1j * H(0,cXX+delXX,cYY+delYY,delXI+pulse_data.iat[i,1],delIX+pulse_data.iat[i,2],delYI+pulse_data.iat[i,3],delIY+pulse_data.iat[i,4]) * t0).expm()*U
    return U

#Infidelty calculation
def inF(x,y):
    return 1-((1/16) * ((x.dag()*y).tr()) * np.conjugate(((x.dag()*y).tr())))

# Below defines the noise in single qubit parameters, due to a single dipole, for a dot whose center lies at (x,0) as a function of dipole center of mass position (xi,yi,h)
# and orientation in the interface plane p. l(nm) is the characteristic dot length, D(nm) is the distance from dot to interface, d(nm) is the dipole length. xi(nm), yi(nm), x(nm)
noisecoef=10**9*e**2/(2*h*4*np.pi*eps0*epsr) #(Hz nm)
def fXnoise(x,l,D,d,xi,yi,p):
    return noisecoef*d*l**2*(3*(x-xi)*(-2*D**2+3*(x-xi)**2-7*yi**2)*np.cos(p)+3*yi*(-2*D**2-7*(x-xi)**2+3*yi**2)*np.sin(p))/(D**2+(x-xi)**2+yi**2)**(7/2)
def fYnoise(x,l,D,d,xi,yi,p):
    return 2*noisecoef*d*l**2*(3*yi*(D**2-4*(x-xi)**2+yi**2)*np.cos(p)-3*(x-xi)*(D**2+(x-xi)**2-4*yi**2)*np.sin(p))/(D**2+(x-xi)**2+yi**2)**(7/2)

# Below quantifies how much a quantum dot whose center lies at (x,0) shifts as a function of dipole center of mass position (xi,yi,h)
# and orientation in the interface plane p.
shiftcoef=10**(-9)*m*e**2/(hb**2*4*np.pi*eps0*epsr) #(1/nm)
#below is the shift function. It tells you how much your quantum dot has shifted by due to a dipole at xi,yi,p in nm meters as a function
#dot size l, distance from interface of dot h, and position of dot along x axis x.
def shift(x,l,D,d,xi,yi,p):
    return (shiftcoef*d*l**4*((D**2-2*(x-xi)**2+yi**2)*np.cos(p)+3*(x-xi)*yi*np.sin(p)))/(D**2+(x-xi)**2+yi**2)**(5/2)

#below is the change in interdot distance units of nm
def Lchange(dx1,dx2):
    return dx2-dx1

# Derivative of the coupling coefficents with respect to interdot distance
dfxxL=-2.16994 * 10**8 #Hz/nm at 63 nm dot sep and l =10nm
dfyyL=1.69414 * 10**8 #Hz/nm at 63 nm dot sep and l = 10nm
# Derivative of monopole term with respect to interdot distance
daL=-1.04256 * 10**9 #Hz/nm at 63 nm dot sep and l = 10nm

# Fucntion below calculates the perturbation in single qubit and two qubit contorl parameters due to a particular, randomly generated instance of dipoles
# x1 is on the left with and x2 is on the right. A(nm**2)
# 10**(-9) in final noise values is to convert ot GHz
def noise(n,A,d,l,D,L):
    x1=-L/2
    x2=L/2
    for a in range(1,n+1):
        if a==1:
            XInoise=0
            IXnoise=0
            YInoise=0
            IYnoise=0
            XXnoise=0
            YYnoise=0
            x1shift=0
            x2shift=0
        xrand=np.random.uniform(-(1/2)*np.sqrt(A),(1/2)*np.sqrt(A))
        yrand=np.random.uniform(-(1/2)*np.sqrt(A),(1/2)*np.sqrt(A))
        prand=np.random.uniform(0,2*np.pi)
        XInoise = XInoise + fXnoise(x1,l,D,d,xrand,yrand,prand)
        IXnoise = IXnoise + fXnoise(x2,l,D,d,xrand,yrand,prand)
        YInoise = YInoise + fYnoise(x1,l,D,d,xrand,yrand,prand)
        IYnoise = IYnoise + fYnoise(x2,l,D,d,xrand,yrand,prand)
        x1shift = x1shift + shift(x1,l,D,d,xrand,yrand,prand)
        x2shift = x2shift + shift(x2,l,D,d,xrand,yrand,prand)
    delL=Lchange(x1shift,x2shift)
    XXnoise = delL*dfxxL
    YYnoise = delL*dfyyL
    XInoise = XInoise + delL*daL
    IXnoise = IXnoise + delL*daL
    list=[XInoise*10**(-9),IXnoise*10**(-9),YInoise*10**(-9),IYnoise*10**(-9),XXnoise*10**(-9),YYnoise*10**(-9),delL*10**3,x1shift*10**3,x2shift*10**3]
    return list

# Below is a function which calcualtes the average perturbation in single and two qubit control parameters for N possible dipole ensemble configurations
def avg_noise_strength(N,n,A,d,l,D,L):
    for i in range(0,N):
        if i == 0:
            XInoise=[]
            IXnoise=[]
            YInoise=[]
            IYnoise=[]
            XXnoise=[]
            YYnoise=[]
            delL_noise=[]
            x1_noise=[]
            x2_noise=[]
        nose_1=noise(n,A,d,l,D,L)
        XInoise.append(nose_1[0])
        IXnoise.append(nose_1[1])
        YInoise.append(nose_1[2])
        IYnoise.append(nose_1[3])
        XXnoise.append(nose_1[4])
        YYnoise.append(nose_1[5])
        delL_noise.append(nose_1[6])
        x1_noise.append(nose_1[7])
        x2_noise.append(nose_1[8])
    avg_noise_list=[sum(XInoise)/N,sum(IXnoise)/N,sum(YInoise)/N,sum(IYnoise)/N,sum(XXnoise)/N,sum(YYnoise)/N,sum(delL_noise)/N,sum(x1_noise)/N,sum(x2_noise)/N]
    noise_standard_dev=[np.sqrt(np.var(XInoise)),np.sqrt(np.var(IXnoise)),np.sqrt(np.var(YInoise)),np.sqrt(np.var(IYnoise)),np.sqrt(np.var(XXnoise)),np.sqrt(np.var(YYnoise)),np.sqrt(np.var(delL_noise)),np.sqrt(np.var(x1_noise)),np.sqrt(np.var(x2_noise))]
    return [avg_noise_list,noise_standard_dev]

# The function below performs the Monte Carlo simulation for a specific pulse
def monte(fZ,cXX,cYY,N,n,A,d,l,D,L,gate,pulse_data):
    for i in range(0, N):
        if i==0:
            tot_inf=0
        nose=noise(n,A,d,l,D,L)
        U_fin=final_U(fZ,cXX,cYY,nose[0],nose[1],nose[2],nose[3],nose[4],nose[5],pulse_data)
        tot_inf=tot_inf+inF(U_fin,gate)
    return tot_inf/N

# Given a list of all pulses and target gates, this computes the Monte Carlo simluation for each pulse and returns the average infidelity in an array.
def monte_all(fz,cXX,cYY,N,n,A,d,l,D,L,gate_list,pulse_data_list):
    for i in range(0,len(gate_list)):
        if i ==0:
            fid_list=[]
        fid_list.append(monte(fz,cXX,cYY,N,n,A,d,l,D,L,gate_list[i],pulse_data_list[i]))
    return fid_list

# This is simply a check that, without noise, the pulse you read in are reaching the target gate with the infidelity you expect.
def no_noise_inF(fZ,cXX,cYY,gate_list,pulse_data_list):
    for i in range(0, len(pulse_data_list)):
        if i==0:
            inf_list_1=[]
        U_fin=final_U(fZ,cXX,cYY,0,0,0,0,0,0,pulse_data_list[i])
        inf_list_1.append(inF(U_fin,gate_list[i]))
    return inf_list_1

# Define all physical parameters in the model
# Zeeman freq (GHz)
fz=4.5
# number of dipole configurations explored in Monte Carlo
N=10**3
# number of dipoles generated in each instance
n=10
# generating interface area (nm**2)
A=10**4
# dipole length (nm)
d=.1
# quantum dot characteristic length (nm)
l=10
# dot to interface distance (nm)
D=100
# inter-dot distance (nm)
L=63
# Two qubit parameters are for L=63 nm and l0= 10nm
# units of (GHz)
Oxx=2.52681
Oyy=-2.02687

# # Test if noise generating function is working
# t1=time.time()
# print(noise(10,10**4,.1,10,100,63))
# t2=time.time()
# print("noise run time")
# print(t2-t1)

# # Test if Unitary evolution function and infidelity calculation is working
# time1=time.time()
# U_fin=final_U(4.5,Oxx,Oyy,0,0,0,0,0,0,BSWAP_pulse_B)
# INF=inF(U_fin,BSWAP1)
# time2=time.time()
# run_time=time2-time1
# print("run time for final gate computation")
# print(run_time)
# print("final gate")
# print(U_fin)
# print("infidelity")
# print(inF(U_fin,BSWAP1))
# print("data length")
# print(len(BSWAP_pulse_B))
# print("average infidelity")
# time21=time.time()
# print(monte(4.5,Oxx,Oyy,10**3,10,10**4,.1,10,100,63,BSWAP1,BSWAP_pulse_B))
# time22=time.time()
# print("average infidelity calculation run time")
# print(time22-time21)

# create a list of different sets of pulse data, and then run through each and tabulate the average infidelity for each

# # Get all pulse infidelities without noise
# print(no_noise_inF(4.5,Oxx,Oyy,gate_list,pulse_data_list))

# average noise strength and standard dev for each of single and two qubit noises
print(avg_noise_strength(10**4,10,10**4,.1,10,100,63))

# # Compute the average infidelity for each pulse and return in array.
# time1=time.time()
# print(monte_all(fz,Oxx,Oyy,10**3,10,10**4,.1,10,100,63,gate_list,pulse_data_list))
# time2=time.time()
# print(time2-time1)

