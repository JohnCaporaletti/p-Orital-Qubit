# p-Orital-Qubit
Code repository for paper entitled "Proposed Five-Electron Charge Quadrupole Qubit" by J. H. Caporaletti and J. P. Kestner

Introduction: This code aims to do three things, 1) calcualte inhomogenous dephasing times for a single p orbital qubit, 2) optimize control pulses for two p orbital qubits, 3) caclculate the average infidelity of the optimize pulses due to charge noise.
Below is how to run/install code for each of these three goals.

1) 
2) Control pulses are found using the qutip-qtrl package which can be downloaded [here] (https://qutip-qtrl.readthedocs.io/en/stable/installation.html). The modules "fidcomp.py", "optimizer.py", "pulseoptim.py", and "optimresult.py" in the qutip-qtrl
   package should be replaced with the corresponding modules in the repository. After this is done, simply download the script "Pulse Optimization w GRAPE.py". Before running the code, you must make two decisions; the Zeeman splitting frequency fz and the
   high frequency cutoff for the bandwidth component of the cost function (inclusive). Finally, "labeling definitions" must be changed to accuratley reflect the optimization being run. The code will pick a random inital condition and perform GRAPE until a
   termination condition is reached by the optimizer. If the termination condition is not "Goal achieved", it will run the optimizer again from a different inital condition. This loop is performed until tot_err < fid_err_targ.
3) 
   
