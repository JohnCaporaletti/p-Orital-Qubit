# p-Orital-Qubit
Code repository for paper entitled "Proposed Five-Electron Charge Quadrupole Qubit" by J. H. Caporaletti and J. P. Kestner

Introduction: This code aims to do three things, 1) calcualte inhomogenous dephasing times for a single p orbital qubit, 2) optimize control pulses for two p orbital qubits, 3) caclculate the average infidelity of the optimize pulses due to charge noise. There are three folders labeled numerically. Each folder has a README that helps understand the code and it's relation to the paper.

3) This is a python script that performs a Monte Carlo simulation on a given set of pulses defined in the script's preamble. You will have to re-write the import statements to access your own GRAPE optimizer data or you can use the papers specific pulse data located in the repositories "pulse_data" folder. The simulation calculates the average pulse infidelity under the presence of a phenomenological dipole two level fluctuator noise model. In addition to this, the script also has other functions that characterize the noise and aid in debugging. Comments in the script guide you through these extra features and example use of them is given at the end of the script but commented out.
   
