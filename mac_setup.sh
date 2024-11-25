# Initialize conda in the shell
eval "$(conda shell.zsh hook)"

# cleaning pip and conda cache
conda clean --all -y
pip cache purge

# Create a new conda environment with Python 3.10
conda create --name llm -y

# Activate the environment
conda activate llm

# Install TensorFlow and other related packages
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
conda install conda-forge::tiktoken -y
conda install conda-forge::jupyter -y
# python -m pip install tensorflow-macos==2.16.2
# python -m pip install tensorflow-metal==1.1.0
# python -m pip install tf-keras==2.16.0

# # Install PyTorch and related packages
# conda install pytorch::pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 -c pytorch -y

# # Install tokenizers and transformers from conda-forge
# conda install conda-forge::datasets=2.21.0 -y
# conda install conda-forge::tokenizers=0.19.1 -y
# conda install conda-forge::transformers=4.44.0 -y
# conda install conda-forge::matplotlib=3.9.2 -y
# conda install conda-forge::tiktoken=0.7.0 -y