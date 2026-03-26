## Isopleth

Isopleth is a contrastive learning model that maps labelling patterns and chemical structures into a shared embedding space, placing labelling patterns for each metabolite near their corresponding structures while pushing unpaired patterns and structures apart. After model training, candidate structures for unidentified peaks are ranked by their cosine similarity to the corresponding labelling patterns for those peaks within this learned embedding space.

This repository contains source code used to train and evaluate Isopleth. `train.py` trains a contrastive learning model and makes predictions on a set of withheld peaks. `evaluate.py` quantifies the performance of that model in the held-out test set using various metrics (e.g. top-k accuracy, top-1 Tanimoto coefficient to the ground-truth structure).

Predictions for all 6,651 peaks (the subset of all 8,277 peaks with at least one candidate structure; predictions made in cross-validation) are provided in `data/predictions.csv.gz`. 

The experiments described in the manuscript were carried out in a conda environment with the following packages installed:

```
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main
_openmp_mutex             5.1                       1_gnu
alabaster                 0.7.16                   pypi_0    pypi
appdirs                   1.4.4                    pypi_0    pypi
attrs                     23.2.0                   pypi_0    pypi
babel                     2.15.0                   pypi_0    pypi
backports-tarfile         1.2.0                    pypi_0    pypi
build                     1.2.1                    pypi_0    pypi
bzip2                     1.0.8                h5eee18b_6
ca-certificates           2024.3.11            h06a4308_0
certifi                   2024.6.2                 pypi_0    pypi
cffi                      1.16.0                   pypi_0    pypi
cfgv                      3.4.0                    pypi_0    pypi
charset-normalizer        3.3.2                    pypi_0    pypi
configargparse            1.7                      pypi_0    pypi
connection-pool           0.0.3                    pypi_0    pypi
contourpy                 1.2.1                    pypi_0    pypi
coverage                  7.5.3                    pypi_0    pypi
coveralls                 4.0.1                    pypi_0    pypi
cryptography              42.0.8                   pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
datrie                    0.8.2                    pypi_0    pypi
deepsmiles                1.0.1                    pypi_0    pypi
distlib                   0.3.8                    pypi_0    pypi
docopt                    0.6.2                    pypi_0    pypi
docutils                  0.20.1                   pypi_0    pypi
dpath                     2.2.0                    pypi_0    pypi
exceptiongroup            1.2.1                    pypi_0    pypi
fastjsonschema            2.20.0                   pypi_0    pypi
fcd-torch                 1.0.7                    pypi_0    pypi
filelock                  3.15.1                   pypi_0    pypi
fonttools                 4.53.0                   pypi_0    pypi
fsspec                    2024.6.0                 pypi_0    pypi
gitdb                     4.0.11                   pypi_0    pypi
gitpython                 3.1.43                   pypi_0    pypi
humanfriendly             10.0                     pypi_0    pypi
identify                  2.5.36                   pypi_0    pypi
idna                      3.7                      pypi_0    pypi
imagesize                 1.4.1                    pypi_0    pypi
importlib-metadata        7.1.0                    pypi_0    pypi
iniconfig                 2.0.0                    pypi_0    pypi
jaraco-classes            3.4.0                    pypi_0    pypi
jaraco-context            5.3.0                    pypi_0    pypi
jaraco-functools          4.0.1                    pypi_0    pypi
jeepney                   0.8.0                    pypi_0    pypi
jinja2                    3.1.4                    pypi_0    pypi
joblib                    1.4.2                    pypi_0    pypi
jsonschema                4.22.0                   pypi_0    pypi
jsonschema-specifications 2023.12.1                pypi_0    pypi
jupyter-core              5.7.2                    pypi_0    pypi
keyring                   25.2.1                   pypi_0    pypi
kiwisolver                1.4.5                    pypi_0    pypi
latexcodec                3.0.0                    pypi_0    pypi
ld_impl_linux-64          2.38                 h1181459_1
libffi                    3.4.4                h6a678d5_1
libgcc-ng                 11.2.0               h1234567_1
libgomp                   11.2.0               h1234567_1
libstdcxx-ng              11.2.0               h1234567_1
libuuid                   1.41.5               h5eee18b_0
markdown-it-py            3.0.0                    pypi_0    pypi

markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.9.0                    pypi_0    pypi
mdit-py-plugins           0.4.1                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
more-itertools            10.3.0                   pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
myst-parser               3.0.1                    pypi_0    pypi
nbformat                  5.10.4                   pypi_0    pypi
ncurses                   6.4                  h6a678d5_0
networkx                  3.3                      pypi_0    pypi
nh3                       0.2.17                   pypi_0    pypi
nodeenv                   1.9.1                    pypi_0    pypi
numpy                     1.26.4                   pypi_0    pypi
nvidia-cublas-cu12        12.1.3.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.1.105                 pypi_0    pypi
nvidia-cudnn-cu12         8.9.2.26                 pypi_0    pypi
nvidia-cufft-cu12         11.0.2.54                pypi_0    pypi
nvidia-curand-cu12        10.3.2.106               pypi_0    pypi
nvidia-cusolver-cu12      11.4.5.107               pypi_0    pypi
nvidia-cusparse-cu12      12.1.0.106               pypi_0    pypi
nvidia-nccl-cu12          2.20.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.5.40                  pypi_0    pypi
nvidia-nvtx-cu12          12.1.105                 pypi_0    pypi
openssl                   3.0.14               h5eee18b_0
packaging                 24.1                     pypi_0    pypi
pandas                    2.2.2                    pypi_0    pypi
pillow                    10.3.0                   pypi_0    pypi
pip                       24.0            py310h06a4308_0
pkginfo                   1.11.1                   pypi_0    pypi
plac                      1.4.3                    pypi_0    pypi
platformdirs              4.2.2                    pypi_0    pypi
pluggy                    1.5.0                    pypi_0    pypi
pre-commit                3.7.1                    pypi_0    pypi
psutil                    5.9.8                    pypi_0    pypi
pulp                      2.7.0                    pypi_0    pypi
pybtex                    0.24.0                   pypi_0    pypi
pybtex-docutils           1.0.3                    pypi_0    pypi
pycparser                 2.22                     pypi_0    pypi
pygments                  2.18.0                   pypi_0    pypi
pyparsing                 3.1.2                    pypi_0    pypi
pyproject-hooks           1.1.0                    pypi_0    pypi
pytest                    8.2.2                    pypi_0    pypi
python                    3.10.14              h955ad1f_1
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2024.1                   pypi_0    pypi
pyyaml                    6.0.1                    pypi_0    pypi
rdkit                     2023.9.6                 pypi_0    pypi
readline                  8.2                  h5eee18b_0
readme-renderer           43.0                     pypi_0    pypi
referencing               0.35.1                   pypi_0    pypi
requests                  2.32.3                   pypi_0    pypi
requests-toolbelt         1.0.0                    pypi_0    pypi
reretry                   0.11.8                   pypi_0    pypi
rfc3986                   2.0.0                    pypi_0    pypi
rich                      13.7.1                   pypi_0    pypi
rpds-py                   0.18.1                   pypi_0    pypi
scikit-learn              1.5.0                    pypi_0    pypi
scipy                     1.11.1                   pypi_0    pypi
seaborn                   0.13.2                   pypi_0    pypi
secretstorage             3.3.3                    pypi_0    pypi
selfies                   2.1.1                    pypi_0    pypi
setuptools                69.5.1          py310h06a4308_0
six                       1.16.0                   pypi_0    pypi
skinniderlab-clm          0.0.2.dev236+g60464be          pypi_0    pypi
smart-open                7.0.4                    pypi_0    pypi
smmap                     5.0.1                    pypi_0    pypi
snakemake                 7.32.4                   pypi_0    pypi
snowballstemmer           2.2.0                    pypi_0    pypi
sphinx                    7.3.7                    pypi_0    pypi
sphinx-rtd-theme          2.0.0                    pypi_0    pypi
sphinxcontrib-applehelp   1.0.8                    pypi_0    pypi
sphinxcontrib-bibtex      2.6.2                    pypi_0    pypi
sphinxcontrib-devhelp     1.0.6                    pypi_0    pypi
sphinxcontrib-htmlhelp    2.0.5                    pypi_0    pypi
sphinxcontrib-jquery      4.1                      pypi_0    pypi
sphinxcontrib-jsmath      1.0.1                    pypi_0    pypi
sphinxcontrib-qthelp      1.0.7                    pypi_0    pypi
sphinxcontrib-serializinghtml 1.1.10                   pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0
stopit                    1.1.2                    pypi_0    pypi
sympy                     1.12.1                   pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
threadpoolctl             3.5.0                    pypi_0    pypi
throttler                 1.2.2                    pypi_0    pypi
tk                        8.6.14               h39e8969_0
tomli                     2.0.1                    pypi_0    pypi
toposort                  1.10                     pypi_0    pypi
torch                     2.3.1                    pypi_0    pypi
tqdm                      4.66.4                   pypi_0    pypi
traitlets                 5.14.3                   pypi_0    pypi
triton                    2.3.1                    pypi_0    pypi
twine                     5.1.0                    pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
tzdata                    2024.1                   pypi_0    pypi
urllib3                   2.2.2                    pypi_0    pypi
virtualenv                20.26.2                  pypi_0    pypi
wheel                     0.43.0          py310h06a4308_0
wrapt                     1.16.0                   pypi_0    pypi
xz                        5.4.6                h5eee18b_1
yte                       1.5.4                    pypi_0    pypi
zipp                      3.19.2                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_1
```