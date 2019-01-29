# Setup instructions

Clone or download this repository (available at [github.com/fons-/grid-analysis](https://github.com/fons-/grid-analysis)) to get started!

## Source code

Most relevant code resides in `src/` and in `notebooks/`.

`src/` contains general class definitions and methods, which will be used in Notebooks for analysis and visualisation. Notebooks can be viewed by running Jupyter Notebook, as explained below, or by viewing the files in your browser, at [github.com/fons-/grid-analysis/notebooks](https://github.com/fons-/grid-analysis/notebooks).

## Installing Python packages

Make sure Python 3 (tested on 3.7) is installed. When using Windows, Python 3 should be [added to your PATH](https://docs.python.org/3/using/windows.html#using-on-windows).

Open a terminal in the root of the repository. Let's create a virtual environment and install the required packages:

*Unix:*

```bash
python3 -m venv ./venv
source venv/bin/activate
# 'python' will now be mapped to python3
python -m pip install -r requirements.txt
```

*Windows:* (this can also be done using the Visual Studio GUI)

```dos
python -m venv .\venv
venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```



We can exit the virtual environment using the `deactivate` command (Unix & Windows).

To use the PyPSA package, we need a *linear optimisation solver*. Follow [these instructions](https://pypsa.org/doc/installation.html#getting-a-solver-for-linear-optimisation) to install one on your system.

To see a geographical map overlay, follow [these instructions](https://matplotlib.org/basemap/users/installing.html) to install *basemap*. (Could be skipped)



## Jupyter Notebook

To use our virtual environment inside Jupyter Notebook, we need to install it as a *Kernel*. First activate the virtual environment, then run:

```
ipython kernel install --user --name=venv
```

### Running Jupyter Notebook

In the root directory, run:

```
jupyter notebook
```

Follow the instructions printed in the console to open Jupyter.

When you open a Python Notebook, you should now be able to select the virtual environment using Kernel > Change Kernel > venv.

## Visual Studio

Open `grid-analysis.sln` in Visual Studio 2017 to get started. Make sure that the virtual environment is listed under "Python Environments" in the Solution Explorer. If possible, Activate the environment and Open its Interactive Environment. When viewing a `.py` file, you can press Ctrl+Enter to run a code block or selection. 

