# ACESS-OM2 models run files and analysis code 

### Mean state perturbed model 
1. calculate the bias at each grid cell at each time step 

$$
\text{bias}_{i,j} = A_{i,j} - \frac{1}{T} \sum^{T}_{t=1} \tau_{i,j}(t)
$$
- $\tau$ is the wind stress at each grid cell in time
- $T$ is the total number of time states
- $A$ is the mean state gained from  data
- 
At each time step, new adjusted wind stress/speed $\tau'$ is found by adding bias to the to each grid cell 
$$
\tau'_{i,j} = \tau_{i,j}(t) + \text{bias}_{i,j}
$$
### Trend peturbed model 

### Analysis code 


