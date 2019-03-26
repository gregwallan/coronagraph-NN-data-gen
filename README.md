These instructions assume you already have a version of python 3.6+ and Jupyter. They were written for use on a Mac.
To set up with jupyter:
First create a virtual environment "nn-gen-venv", run the following in this repo's directory.
`python3 -m venv nn-gen-venv`

Then activate the environment:
`source nn-gen-venv/bin/activate`

Then install dependencies:
`pip install -r requirements.txt`

Then register the environment's python executable in jupyter/ipython:
`ipython kernel install --user --name=nn-gen-venv`

You can launch jupyter with:
`jupyter notebook`

Then open the desired notebook. In the menu bar click 
`Kernel>Change kernel>nn-gen-venv`
