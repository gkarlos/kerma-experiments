# Count Benchmark Accesses (Pilot)

In this experiment we analyze the memory accesses performed by the kernels of the Rodinia benchmark
in order to determine for how many of those we can infer their location statically, given knowledge of:
- Grid and Block dimensions
- Block and thread indices
- Values for kernel scalar arguments

## Build and run
TODO