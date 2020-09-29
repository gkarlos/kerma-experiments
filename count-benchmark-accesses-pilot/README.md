# Count Benchmark Accesses (Pilot)

In this experiment we analyze the memory accesses performed by the kernels of the Rodinia benchmark
in order to determine for how many of those we can infer their location statically, given knowledge of:
- Grid and Block dimensions
- Block and thread indices
- Values for kernel scalar arguments

## Build and run
```
python3 -m venv .env
source .env/bin/activate
make plots
```

## Rodinia
### Selected
- backprop
- bfs
- cfd<sup id="a1">[1](#f1)</sup>
- gaussian
- hotspot
- hybridsort
- kmeans
- lavaMD
- lud
- nn (+check IR, +maybe fix manually)
- nw
- particlefilter
- pathfinder
- srad
- streamcluster

### Not selected
- b+tree (complex struct arguments)
- dwt2d (c++ templates)
- heartwall (links with avilib)
- mummergpu (complex args + multifile + liblinking)
- leukocyte

### Fix manually (maybe)
- myocyte

<a id="f1"><b>[1]</b></a> only using euler3d.cu the rest are basically identical except for using `double` instead of `float` types.[↩](#a1)