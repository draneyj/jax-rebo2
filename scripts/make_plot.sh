
module load anaconda3/2025.6
module load intel-oneapi/2024.2; module load intel-mpi/oneapi/2021.13; module load intel-mkl/2024.2; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jd6157/.local/lib64
conda activate jaxmd

shopt -s extglob

param_files=""
for prefix in  all_4_reg2p0 all_5_reg2p0 all_4_reg2p0L; do 
    param_files="${param_files} $(ls -t params/${prefix}_+([0-9])_params.pkl | head -1)"
done
echo $param_files
JAX_PLATFORMS=cpu python pair_force.py -p $param_files -r -o pair.png
code pair.png
