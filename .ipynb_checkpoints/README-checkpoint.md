



## Mean state

1. calculate the bias at each grid cell at each time step 

$$
\text{bias}_{i,j} = A_{i,j} - \frac{1}{T} \sum^{T}_{t=1} \tau_{i,j}(t)
$$
- $\tau$ is the wind stress at each grid cell in time
- $T$ is the total number of time states
- $A$ is the mean state gained from scratterometer data 

2. Adjust the wind stress/ speed fields

At each time step, new adjusted wind stress/speed $\tau'$ is found by adding bias to the to each grid cell 
$$
\tau'_{i,j} = \tau_{i,j}(t) + \text{bias}_{i,j}
$$

## Trend adjustments

Let $A_{i,j}$ be the slope of the linear trend of wind stress/speed $\tau_{i,j}$, where $A^s_{i,j}$ is the slope of the scratterometer data and $A'$ is the adjusted slope of JRA55-DO wind stress/speed.
For JRA55-DO, linear fit of wind stress can be given as: 
$$
\tau_{i,j}(t) = m_{i,j}t + c_{i,j}
$$
For scratterometer the trend is given by $M$, then adjusted wind stress/speed can be given by: 
$$
\tau'_{i,j}(t) = \tau_{i,j}(t) - m_{i,j} t + M_{i,j} t + b_{i,j}
$$



