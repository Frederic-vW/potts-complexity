# potts-complexity

- The functions for entropy rate, excess entropy, Lempel-Ziv complexity (LZ-76), and DFA for discrete time series are contained in the file `potts_complexity.py`.  
- The figure below shows a complexity analysis of sample data (n=10 lattice sites) from Potts model simulations (2D lattice) and can be reproduced with the notebook `potts_complexity.ipynb`.  
- You can run the notebook on mybinder by clicking the "launch binder" icon below:  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Frederic-vW/potts-complexity/main?labpath=potts_complexity.ipynb)

<!--
![Fig_potts](Fig_potts_complexity.png)
-->
<img src="Fig_potts_complexity.png" width="600"/>

### Potts 2D visualization
Dynamics of the 2D Potts system (Q=5) on a lattice (128 x 128)  are shown below. The dynamics from the initial random state are included. The critical temperature is Tc=0.85.

Low temperature, T=0.51 (0.6 Tc)
<p align="center">
<video src="videos/Potts2D_Q5_T0.51_L128_N1000.webm" width="256" height="256" controls preload></video>
</p>

Critical temperature, T=0.85 (1.0 Tc)
<p align="center">
<video src="videos/Potts2D_Q5_T0.85_L128_N1000.webm" width="256" height="256" controls preload></video>
</p>

High temperature, T=2.55 (3.0 Tc)
<p align="center">
<video src="videos/Potts2D_Q5_T2.55_L128_N1000.webm" width="256" height="256" controls preload></video>
</p>
