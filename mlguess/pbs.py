import os
import subprocess
import yaml


def launch_pbs_jobs(config_file, trainer_path, args = ''):
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
    #PBS -l gpu_type={config['pbs']['gpu_type']}
    #PBS -A {config['pbs']['account']}
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
    print(jobid)

    # Clean up PBS script
    os.remove("launcher.sh")
