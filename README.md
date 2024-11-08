# p-Orital-Qubit
Code repository for paper entitled "Proposed Five-Electron Charge Quadrupole Qubit" by J. H. Caporaletti and J. P. Kestner

Introduction: This code aims to do three things, 1) calcualte inhomogenous dephasing times for a single p orbital qubit, 2) optimize control pulses for two p orbital qubits, 3) caclculate the average infidelity of the optimize pulses due to charge noise.
Below is how to run/install code for each of these three goals.

1) This file is a mathematica notebook. There is no special requirements or downloads required to run it. It simply generates inhomogenous dephasing time data and allows you to export it.
2) Control pulses are found using the qutip-qtrl package which can be downloaded [here](https://qutip-qtrl.readthedocs.io/en/stable/installation.html). The modules "fidcomp.py", "optimizer.py", "pulseoptim.py", and "optimresult.py" in the qutip-qtrl
   package should be replaced with the corresponding modules in the repository. After this is done, simply download the script "Pulse Optimization w GRAPE.py". Before running the code, you must make two decisions; the Zeeman splitting frequency fz and the
   high frequency cutoff for the bandwidth component of the cost function (inclusive). Finally, "labeling definitions" must be changed to accuratley reflect the optimization being run. The code will pick a random inital condition and perform GRAPE until a
   termination condition is reached by the optimizer. If the termination condition is not "Goal achieved", it will run the optimizer again from a different inital condition. This loop is performed until tot_err < fid_err_targ.
3) This is a python script that performs a Monte Carlo simulation on a given set of pulses defined in the script's preamble. You will have to re-write the import statements to access your own GRAPE optimizer data or you can use the papers specific pulse data located in the repositories "pulse_data" folder. The simulation calculates the average pulse infidelity under the presence of a phenomenological dipole two level fluctuator noise model. In addition to this, the script also has other functions that characterize the noise and aid in debugging. Comments in the script guide you through these extra features and example use of them is given at the end of the script but commented out.
   
