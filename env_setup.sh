#!/bin/bash

# Script to install required packages in conda

BOLDYELLOW='\e[1;33m'
BOLDGREEN='\e[1;32m'
ENDC='\e[0m'

if { conda env list | grep ".*dini*"; } >/dev/null 2>&1
then
	conda activate dini
	echo -e "${BOLDGREEN}Environment already installed!${ENDC}"
else
	if [ "$(uname)" = "Darwin" ]; then
		# Mac OS X platform
		# Conda can be installed from here - https://github.com/conda-forge/miniforge
		echo -e "${BOLDYELLOW}Platform discovered: macOS${ENDC}"

		# Rust needs to be installed
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

		# Install environment
		conda create --name dini python=3.9

		# Activate environment
		conda activate dini

		# Add basic packages and enabling extentions
		conda install -c conda-forge tqdm ipywidgets matplotlib scikit-optimize
		jupyter nbextension enable --py widgetsnbextension
		conda install -c anaconda scipy cython
		conda install pyyaml
		conda install pandas

		# Install PyTorch and PyTorch-Geometric for GRAPE. Tested on Apple M1 Pro
		conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
		MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch torchvision torchaudio
		MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html
		MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+${cpu}.html
		MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-geometric

		# Install fancyimpute (noarch version for Apple M1)
		conda install -c conda-forge fancyimpute

		# Conda prefers pip packages to be installed in the end
		pip install sko
	else
		echo -e "${BOLDYELLOW}Platform discovered: Linux/Windows. Use 'requirements.txt' file.${ENDC}"
	fi
fi
