# covid19-challenge

This repository contains information to tackle https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

How to prepare my environment to start to collaborate ??

* Install Git Bash
  https://git-scm.com/downloads
  
* Install Anaconda

  Download Anaconda 3.7 for your platform and install it.
  https://www.anaconda.com/distribution/
 
 * Install Visual Studio Code
 
  Download and install it from https://code.visualstudio.com/download
  
 * Clone this repository
 
  From the Git Bash console, run
  
 ```bash
 git clone https://github.com/ezequielma20/covid19-challenge.git
 ```
 
 * Run an Anaconda Prompt
 * Create the environment with:
 
 ```bash 
 conda env update --name nlp --file environment.yml
 ```

 (or this one if any error occurs
 
 ```
 conda env update --prefix ./env --file environment.yml 
 ```

 )
 
  * Activate the newly created environment
  ```bash
  conda activate nlp
  ```
  
  * Export the environmen if you add a new library
  ```bash
  conda env export -n nlp --from-history > environment.yml
  ```
  
# Referencies

* https://spacy.io/usage/models
