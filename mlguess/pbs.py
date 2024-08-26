import os
import re
import yaml
import shutil
import logging
import subprocess


def launch_pbs_jobs(config_file, trainer_path, args=''):
    """Launches a PBS job using the specified configuration file and trainer script.

    This function reads the configuration file to construct a PBS script, writes the
    script to a file, submits the job using `qsub`, and then cleans up the script file.

    Args:
        config_file (str): Path to the YAML configuration file containing PBS options.
        trainer_path (str): Path to the Python training script to be executed.
        args (str, optional): Additional command-line arguments to pass to the training script. Defaults to an empty string.

    Raises:
        ValueError: If the 'pbs' section is not present in the configuration file.

    """
    # Load configuration file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if "pbs" not in config:
        raise ValueError(
            "You must add a pbs field to the model configuration. See config/pbs.yml for an example"
        )

    # Build PBS script
    save_path = config["save_loc"]

    script = f"""#!/bin/bash -l
    #PBS -N {config['pbs']['name']}
    #PBS -l select={config['pbs']['select']}:ncpus={config['pbs']['ncpus']}:ngpus={config['pbs']['ngpus']}:mem={config['pbs']['mem']}
    #PBS -l walltime={config['pbs']['walltime']}
    """

    # Add optional fields if they exist
    if 'gpu_type' in config['pbs']:
        script += f"#PBS -l gpu_type={config['pbs']['gpu_type']}\n"
    if 'cpu_type' in config['pbs']:
        script += f"#PBS -l cpu_type={config['pbs']['cpu_type']}\n"

    script += f"""#PBS -A {config['pbs']['account']}
    #PBS -q {config['pbs']['queue']}
    #PBS -o {os.path.join(save_path, "out")}
    #PBS -e {os.path.join(save_path, "out")}
    {config['pbs']['env_setup']}
    python {trainer_path} -c {config_file} {args}
    """

    # Write PBS script to file
    with open("launcher.sh", "w") as fid:
        fid.write(script)

    # Submit PBS job
    jobid = subprocess.Popen(
        "qsub launcher.sh",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()[0]
    jobid = jobid.decode("utf-8").strip("\n")
    logging.info(f"Launched job {jobid}")

    # Clean up PBS script
    os.remove("launcher.sh")


def launch_distributed_jobs(config_file, script_path, launch=True):
    """Launches a distributed job across multiple nodes using PBS and MPI.

    This function generates a PBS script based on the provided configuration file,
    copies the necessary files, and optionally submits the job to the queue.

    Args:
        config_file (str): Path to the YAML configuration file containing PBS options.
        script_path (str): Path to the Python script to be executed in the distributed environment.
        launch (bool, optional): If True, submits the job using `qsub`. If False, only generates the script. Defaults to True.

    """
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

    if os.path.exists(config_save_path):
        os.remove(config_save_path)
        logging.info('Remove the old model.yml at {}'.format(config_save_path))

    shutil.copy(config_file, config_save_path)
    logging.info('Copy the new {} to {}'.format(config_file, config_save_path))

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
        logging.info(jobid)

        # copy launch.sh to the design location
        launch_path = os.path.join(save_loc, "launch.sh")

        if os.path.exists(launch_path):
            os.remove(launch_path)
            print('Remove the old launch.sh at {}'.format(launch_path))

        shutil.copy("launch.sh", os.path.join(save_loc, "launch.sh"))
        logging.info('Generating the new script at {}'.format(launch_path))

        # remove the one from local space
        os.remove("launch.sh")
