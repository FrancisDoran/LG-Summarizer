# Setup Python Environment

This project uses python to initialize a virtual environment management, and pip for package management.  
To set up the python environment follow the instructions below.

``` bash
#!/bin/bash  

cd LG-Summarizer/build/python_deps/  
  
# run init.sh...
#       The first run creates a virtual environment if one isn't present.
#
#       The second run (after doing source venv/bin/activate) installs
#       the dependencies into the virtual environment.
#
#       If you want to install the dependencies into a different virtual environment,
#       you can setup your virtual env, activate it, and then run 
#       ./install_deps.sh or pip install -r requirements.txt
./init.sh  
```

