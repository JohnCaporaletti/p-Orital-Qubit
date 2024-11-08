import numpy as np
# Numpy functions for Fast Fourier Transform
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip_qip.algorithms import qft
import qutip_qtrl.logging_utils as logging
logger =logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level=logging.WARN
#QuTiP control modules
import qutip_qtrl.pulseoptim as cpo
import qutip_qtrl.pulsegen as pulsegen

#define basic quantum objects
X=sigmax()
Y=sigmay()
Z=sigmaz()
I=identity(2)
S=Qobj([[1,0],[0,np.exp(1j*(np.pi/2))]])
H=(1/np.sqrt(2))*Qobj([[1,1],[1,-1]])
T=Qobj([[1,0],[0,np.exp(1j*(np.pi/4))]])
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
CNOT=(1j * (np.pi/4) * tensor(I-Z,I-X)).expm()
BSWAP=(1j * (np.pi/4) * (XX-YY)).expm()

# Below defines optimizer parameters. All frequencies are in units of GHz.
######################################
#Zeeman frequency
fz=4.5
#High frequency cutoff
hfc=5
#Two qubit frequencies for L=63 nm and l0= 10nm (O stands for Omega)
Oxx=2.52681
Oyy=-2.02687
#Drift Hamiltonian
H_d= np.pi * (fz*(IZ+ZI)+Oxx*XX+Oyy*YY)
#Control Hamiltonian definition
if fz!=0:
    H_c=[np.pi*XI,np.pi*IX]
else:
    H_c=[np.pi*XI,np.pi*IX,np.pi*YI,np.pi*IY]
#Number of control parameters
n_ctrls=len(H_c)
#Inital condition for unitary evolution
U_0=II
#Target unitary
U_targ=HI
#Amplitude bounds on each control parameter
amp_ubound=10
amp_lbound=-10
# Duration of each timeslot
dt = 0.006
# List of evolution times to try
evo_time = 1
n_ts = int(float(evo_time) / dt)
# Fidelity error target
fid_err_targ = 10**(-3)
# Maximum iterations for the optisation algorithm
max_iter = 5000
# Maximum (elapsed) time allowed in seconds
max_wall_time = 10000
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-10
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'RND'
################################################

#Below is labeling definitions for saved pulse data file name
###################################################
#Two qubit inputs
XXYY='L=63,l0=10'
#Control parameters
Control='XI+IX+YI+IY'
#Target gate
Gate='HI'

#Set to None to suppress output files
f_ext = "GRAPE_high_freq_cut_off_{}_fz_{}_2Qubit_Coupling_{}_Cntrl_{}_Cntrl_Amps_{}_Target_Gate_{}_evo_time_{}_n_ts_{}_Int_ptype_{}".format(hfc,fz,XXYY,Control,amp_ubound,Gate,evo_time,n_ts,p_type)
###################################################

#Initialize boolean variable and optimizer attempt counter
bool=True
counter=0
while bool:
    counter=counter + 1
    #Define pulse optimizer instance
    optim = cpo.create_pulse_optimizer(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                    amp_lbound=amp_lbound, amp_ubound=amp_ubound, 
                    fid_err_targ=fid_err_targ, min_grad=min_grad, 
                    max_iter=max_iter, max_wall_time=max_wall_time, optim_method='fmin_l_bfgs_b',
                    method_params={'max_metric_corr':40, 'accuracy_factor':10},
                    dyn_type='UNIT', 
                    fid_params={'phase_option':'PSU'},
                    init_pulse_type=p_type, 
                    log_level=log_level, gen_stats=True, hfc=hfc)
    
    #Create instance of dynamics and pulse generator
    optim.test_out_files = 0
    dyn = optim.dynamics
    dyn.test_out_files = 0
    p_gen = optim.pulse_generator

    #Create initial pulse profile
    p_gen = pulsegen.create_pulse_gen(p_type, dyn)
    init_amps = np.zeros([n_ts, n_ctrls])

    if (isinstance(p_gen, pulsegen.PulseGenLinear)):
        for j in range(n_ctrls):
            p_gen.scaling = float(j) - float(n_ctrls - 1)/2
            init_amps[:, j] = p_gen.gen_pulse()
    elif (isinstance(p_gen, pulsegen.PulseGenZero)):
        for j in range(n_ctrls):
            p_gen.offset = sf = float(j) - float(n_ctrls - 1)/2
            init_amps[:, j] = p_gen.gen_pulse()
    else:
        # Should be random pulse
        for j in range(n_ctrls):
            init_amps[:, j] = p_gen.gen_pulse()

    #Tell dynamics instance what the inital control amplitudes are
    dyn.initialize_controls(init_amps)

    # Below can be uncommented so that each random initial seed is saved. WARNING: If the optimizer can't find a solution or takes a very long time, this will cause a large number of files to be saved.
    # # Save initial amplitudes to a text file
    # if f_ext is not None:
    #     pulsefile = "ctrl_amps_initial_" + f_ext + ".txt"
    #     dyn.save_amps(pulsefile)
    #     print("Initial amplitudes output to file: " + pulsefile)

    print("***********************************")
    print("\n+++++++++++++++++++++++++++++++++++")
    print("Starting pulse optimisation attempt # {} for T={}".format(counter, evo_time))
    print("+++++++++++++++++++++++++++++++++++\n")

    #Run optimization!
    result = optim.run_optimization()

    # get handle on results
    result.stats.report()
    #Determine if attempt was successful and terminate loop depending on result
    if result.tot_err <= fid_err_targ:
        bool=False
    else:
        bool=True

# Save final amplitudes to a text file
if f_ext is not None:
    pulsefile = "ctrl_amps_final_" + f_ext + "_Fin_Fid_Err_{}_Wall_Time_{}.txt".format(result.fid_err,result.wall_time)
    dyn.save_amps(pulsefile)
    print("Final amplitudes output to file: " + pulsefile)

print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final high frequency error {}".format(result.hf_err))
print("Final total error {}".format(result.tot_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result.wall_time)))

#Create plot output
fig1 = plt.figure(figsize=(12,8))

#Initial amps plot
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Init amps T={}".format(evo_time))
# ax1.set_xlabel("Time")
ax1.get_xaxis().set_visible(False)
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(result.time, 
            np.hstack((result.initial_amps[:, j], 
                    result.initial_amps[-1, j])), 
                where='post')
    
#Optimised amps plot
ax2 = fig1.add_subplot(2, 1,2)
ax2.set_title(Gate.format(evo_time))
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Control amplitude (GHz)")
for j in range(n_ctrls):
    ax2.step(result.time, 
            np.hstack((result.final_amps[:, j], 
                    result.final_amps[-1, j])), 
                where='post')
        
# Final contrl amplitudes
amps = dyn.ctrl_amps
# Show FFT of result
n_ctrls = dyn.num_ctrls
#number of time steps in the pulse
n_ts = dyn.num_tslots
# sampling rate
sr=n_ts/evo_time
#sampling interval
ts=1/sr
f=rfftfreq(n_ts,ts)

# plot FFT of result
plt.figure(figsize = (12, 6))
for j in range(n_ctrls):
    X=rfft(amps[:,j],norm="forward")
    plt.subplot(int("23" + str(j+1)))
    plt.stem(f,np.sqrt(np.real(X*np.conjugate(X))), 'b', \
        markerfmt=" ", basefmt="-b")
    plt.xlim(0, 20)

#plot weighting function
weight1=dyn.fid_computer.w1(f,hfc)
plt.subplot(235)
plt.stem(f,weight1, 'b', \
    markerfmt=" ", basefmt="-b")
plt.xlim(0, 20)

# Show all plots made so far
plt.tight_layout()
plt.show()