### Scheduling the tasks
We only want to schedule the task on the compute node "desktop" 1 to 10, because those have the same underlying hardware and a single GPU.

*PyTorch*
sbatch -w "desktop[1-10]" run_experiments_GPU_pytorch.job

*Tensorflow*
sbatch -w "desktop[1-10]" run_experiments_GPU_tensorflow.job