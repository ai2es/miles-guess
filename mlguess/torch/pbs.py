import re
import os
import yaml
import shutil
import subprocess


def launch_script(config_file, script_path, launch=True):

    # Load the configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Extract PBS options from the config
    pbs_options = config['pbs']

    config_save_path = os.path.expandvars(os.path.join(config["save_loc"], "model.yml"))

    # Generate the PBS script
    script = f"""#!/bin/bash -l
    #PBS -N {pbs_options['job_name']}
    #PBS -l select=1:ncpus={pbs_options['ncpus']}:ngpus={pbs_options['ngpus']}:mem={pbs_options['mem']}
    #PBS -l walltime={pbs_options['walltime']}
    #PBS -l gpu_type={pbs_options['gpu_type']}
    #PBS -A {pbs_options['project']}
    #PBS -q {pbs_options['queue']}
    #PBS -j oe
    #PBS -k eod

    source ~/.bashrc

    conda activate {pbs_options['conda']}

    python {script_path} -c {config_save_path}
    """

    script = re.sub(r'^\s+', '', script, flags=re.MULTILINE)

    # Save the script to a file
    with open('launch.sh', 'w') as script_file:
        script_file.write(script)

    if launch:
        jobid = subprocess.Popen(
            "qsub launch.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        print(jobid)
        save_loc = os.path.expandvars(config["save_loc"])
        if not os.path.exists(os.path.join(save_loc, "launch.sh")):
            shutil.copy('launch.sh', os.path.join(save_loc, "launch.sh"))
        os.remove("launch.sh")


def launch_script_mpi(config_file, script_path, launch=True):

    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    # Extract PBS options from the config
    pbs_options = config.get('pbs', {})

    user = os.environ.get('USER')
    num_nodes = pbs_options.get('nodes', 1)
    num_gpus = pbs_options.get('ngpus', 1)
    total_gpus = num_nodes * num_gpus

    # Create the CUDA_VISIBLE_DEVICES string
    cuda_devices = ",".join(str(i) for i in range(total_gpus))
    save_loc = os.path.expandvars(config["save_loc"])

    config_save_path = os.path.join(save_loc, "model.yml")

    # Generate the PBS script
    script = f'''#!/bin/bash
    #PBS -A {pbs_options.get('project', 'default_project')}
    #PBS -N {pbs_options.get('job_name', 'default_job')}
    #PBS -l walltime={pbs_options.get('walltime', '00:10:00')}
    #PBS -l select={num_nodes}:ncpus={pbs_options.get('ncpus', 1)}:ngpus={num_gpus}:mem={pbs_options.get('mem', '4GB')}
    #PBS -q {pbs_options.get('queue', 'default_queue')}
    #PBS -j oe
    #PBS -k eod

    # Load modules
    module purge
    module load nvhpc cuda cray-mpich conda
    conda activate {pbs_options.get('conda', 'holodec')}

    # Get a list of allocated nodes
    nodes=( $( cat $PBS_NODEFILE ) )
    head_node=${{nodes[0]}}
    head_node_ip=$(ssh $head_node hostname -i | awk '{{print $1}}')

    # Export environment variables
    export LSCRATCH=/glade/derecho/scratch/{user}/
    export LOGLEVEL=INFO
    export NCCL_DEBUG=INFO

    # Print the results
    echo "Number of nodes: {num_nodes}"
    echo "Number of GPUs per node: {num_gpus}"
    echo "Total number of GPUs: {total_gpus}"

    # Log in to WandB if needed
    # wandb login 02d2b1af00b5df901cb2bee071872de774781520

    # Launch MPIs
    CUDA_VISIBLE_DEVICES="{cuda_devices}" mpiexec -n {num_nodes} --ppn 1 --cpu-bind none torchrun --nnodes={num_nodes} --nproc-per-node={num_gpus} --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip {script_path} -c {config_save_path}
    '''

    script = re.sub(r'^\s+', '', script, flags=re.MULTILINE)

    # Save the script to a file
    with open('launch.sh', 'w') as script_file:
        script_file.write(script)

    if launch:
        jobid = subprocess.Popen(
            "qsub launch.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        print(jobid)
        if not os.path.exists(os.path.join(save_loc, "launch.sh")):
            shutil.copy("launch.sh", os.path.join(save_loc, "launch.sh"))
        os.remove("launch.sh")


if __name__ == "__main__":
    config_file = "../config/vit2d.yml"
    # Where does this script live?
    script_path = "../applications/trainer_vit2d.py"
    launch_script(config_file, script_path, launch=False)
    #launch_script_mpi(config_file, script_path, launch = False)
