## Steps to contribute (generelizes to any GitHub project)
* Have the initial discussion in the issue itself.
* Once you are clear on To Dos, fork this repo. It will create a replica of this repo in your GitHub account.
* Never work on the `main` branch. Create a new branch to solve a new issue.
* After the initial part of work is done, open a PR to this repo and instructors will review the PR.

## Guidelines

* VS code IDE is recommended. Use AI tools like GitHub Co-pilot to speed up your coding. You can get free access if you are a student by following [this tutorial](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal).
* Use `black` as a code formatter. Install using `pip install black[jupyter]`. Enable `Notebook › Format On Save` in VS code setting to format your code automatically on save.
* Use [PEP-8](https://peps.python.org/pep-0008/) code style wherever possible. A few important points are the following:
    * File names and variable names are lower case and words are separated by underscores e.g. `example_1.py`
    * Class names are upper case and words start with upper case e.g. `BayesianLinearRegression`