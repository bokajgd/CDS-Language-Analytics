# W4 - Sentiment Analysis

# Overview 

**Jakob Grøhn Damgaard, March 2021** <br/>
This repository contains the W4 assigmnent for the course *Language Analytics*

# Code
The raw data files can be found in the *data* folder. This folder includes the full data set *abcnews-date-text.csv* and and a subset called *test_data.csv* which contains the first 20000 headlines. <br/>
The code to execute the tasks can be found in the file *W4-Sentiment-Analysis.py*<br/>
The two plots roduced by this script can be found in the folder *output*<br/>

# Download and Execute
To locally download a compressed zip version of this repository, one can zip the entire repository from GitHub by navigating back to the home page of the repository and clicking the *Code* button and then *Download ZIP*. <br/>
<br>
Before executing the .py file, open the terminal, navigate the directory to the folder directory and run the following code to install the requirements list in the *requirements.txt* file in a virtual environment:
<br>
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```
<br>
You can then proceed to run the script in the terminal by running the following line: 
<br>

```bash
python W4-Sentiment-Analysis.py --filename test_data.csv
```

<br>

# License
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

