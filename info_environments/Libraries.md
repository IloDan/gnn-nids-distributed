- numpy
- Pandas
- sklearn
- matplotlib
- cuMl
- pytorch
- FOR GPU P100 Compatibility (compute capability=6.0) 
    - conda create -n KC -c rapidsai -c conda-forge -c nvidia  \
        rapids=23.12 python=3.10 'cuda-version>=12.0,<=12.5' \
        jupyterlab 'pytorch=*=*cuda*'

- lgbm