we are running from this directory: /cluster/home/ivang/TDT4265_Assignment1/assignment4/SSD
 the name of the job is: test
Th job ID is 17672016
The job was run on these nodes: idun-04-01
Number of nodes: 1
We are using 4 cores
We are using 4 cores per node
Total of  cores
Total of GPUS: 0
Tue Mar 28 15:53:32 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.43.04    Driver Version: 515.43.04    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:C0:00.0 Off |                    0 |
| N/A   32C    P0    39W / 250W |      0MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
	[4mGPU0	mlx5_0	mlx5_1	CPU Affinity	NUMA Affinity[0m
GPU0	 X 	SYS	PIX	33,35,37,39	0-1
mlx5_0	SYS	 X 	SYS		
mlx5_1	PIX	SYS	 X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
/var/slurm_spool/job17672016/slurm_script: line 27: cd: /cluter/home/ivang/TDT4265_Assignment1/assignment4/SSD: No such file or directory
Traceback (most recent call last):
  File "train.py", line 7, in <module>
    from omegaconf import OmegaConf
ModuleNotFoundError: No module named 'omegaconf'
