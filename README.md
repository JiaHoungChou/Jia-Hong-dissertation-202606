### *dissertation title: Cross-domain online remaining useful life prediction of lithium-ion battery cells and packs using model weight transfer and unsupervised domain adaptation*

## Package Installation Manual

This section explains how to install the required Python packages for running this project.

### 1. Create a Virtual Environment

It is recommended to create a separate virtual environment before installing the packages.

#### Option 1: Using Conda

```bash
conda create -n myenv python=3.9
conda activate myenv
python -m pip install --upgrade pip
pip install pandas numpy scipy matplotlib scikit-learn EMD-signal torch joblib
'''bash

### CPU version
'''bash
pip install torch
'''bash

## GPU version
'''bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
'''bash
