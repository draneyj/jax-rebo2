# for f in slurm-256310* slurm-2639972.out slurm-2677695.out slurm-2708394.out slurm-2742543.out slurm-2742544.out; do
for f in slurm-3729616.out slurm-3795225.out slurm-3836683.out slurm-3836686.out; do
    echo -en "\033[1;46m"
    echo $f
    grep Name $f
    echo -e "\033[0m"
    grep Step $f | tail -1
    grep best $f -B1 | tail -2 | grep -v best
done