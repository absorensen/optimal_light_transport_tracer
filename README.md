# optimal_light_transport_tracer
Small rust path tracer based on [smallpt](http://www.kevinbeason.com/smallpt/), to better learn and implement optimal light transport algorithms. The project is on the backburner due to shifting priorities. The goal is to eventually implement mixed PDF's, bidirectional path tracing, Metropolis light transport, reSTIR and reSTIR GI on both the CPU and GPU. I created an altered version of the cornell box which should make it impossible for reSTIR and reSTIR GI to converge as it needs at least 3 bounces to even find the second light in the most direct case.

The sample images in the renders folder are all rendered with different for different light transport algorithms, but totaling around 15 seconds of render time.

## Naive

## Dynamic Sampling

## Multiple Importance Sampling
