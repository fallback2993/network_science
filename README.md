# project_network_science

## Experimental Setup
This project contains the code required to reproduce the experiments conducted in the [comparative analysis of different models](./paper/paper_draft.pdf) for the Network Science lecture. The code contains an implementation of the Louvain-Core algorithm as well as implementations for the NMI metric and the Map-Equation.

## Download code
The code will be available as a copy within this zip. However, it is advised to download the most recent code from github.
```console
git clone https://github.com/fallback2993/network_science
cd network_science
```

## Installation
In order to run the experiment, you have to clone and run the jupyter notebook experiment.ipynb. Make sure you have conda or python installed. Then install the requirements.txt dependencies.

```console
pip install -r requirements.txt
jupyter notebook
```

## Running Jupyter
Jupyter will open. Navigate to ./src/experiment.ipynb and open the file. Run all the cell within Jupyter notebook. The code will run for a couple of hours. Furthermore, it will you some of your computational ressources, as it uses multiprocessing to speed up the computation. After completion you will see the plots that were mentioned in the paper and a csv file with the raw data will also be created.

## Additional Info
The last cell will contain the execution of the experiment. There will NOT be any output on the Jupyter visual interface. To see whether the system is still running check the console in which you have started jupyter notebook.
