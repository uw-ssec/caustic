# Caustics Docker Resources

This directory contains the files needed to create a GPU Optimized Docker image
for caustics. Currently, it is optimized for CUDA Version 11.8.0 only.

The following files can be found:

- `Dockerfile`: The docker specification file that contains the instructions on
  how to build the image using docker.
- `env.yaml`: The conda environment yaml file that list out the required
  packages to install for the python conda environment within the docker
  container when spun up.
- `conda-linux-64.lock`: The `linux-64` architecture specific conda lock file
  that specifies the exact urls to the conda packages. This was generated
  manually with the `conda-lock` program using the `env.yaml` file.

  ```bash
  conda-lock -f env.yaml --kind explicit -p linux-64
  ```

- `run-tests.sh`: A really simple script file that is accessible via command
  line within the container by calling `/bin/run-tests`. This file essentially
  runs `pytests`.
