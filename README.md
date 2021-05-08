# Breaking the Transformer Bottleneck
An analysis of the effects of Mixture of Softmaxes on the Transformer NLP architecture.

# Installation Instructions
To make sure that all developers are working with the same tools, please work in a virtual environment.  You can install Python's `venv` [module](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) through the following command (on macOS):
```
python3 -m pip install --user virtualenv
```
Now, create your virtual environment 
```
python3 -m venv env
```
and activate it:
```
source env/bin/activate
```
You can now install packages as you normall would do.  When you are done installing packages, send them to a **`requirements.txt`** file:
```
python3 -m pip freeze > requirements.txt
```
To deactivate your environment, simply run
```
deactivate
```
Now, all the required packages are located in **`requirements.txt`**.  So, when someone wants to download our repository and have their environment replicate ours, they simply need to run
```
python -m pip install -r requirements.txt
```
in their own Python enivornment, whether it be virtual or real.