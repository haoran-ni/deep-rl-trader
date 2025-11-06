# deep-rl-trader
Final project of 10-703 Deep RL @ CMU

# Set up environments

1. Install Anaconda

2. Install OpenAI (required by FinRL)
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
```
3. Create Conda environments
```bash
conda create -n deep-rl-trader python=3.12
pip install -r requirements.txt
```
4. Install `box2d` (not required, we probably do not need physical simulation right now)
```
conda install conda-forge::box2d-py
```
5. Add FinRL as submodule.
Go back to the root directory and run
```
git submodule add https://github.com/jingyan-li/FinRL.git
git submodule update --init --recursive
```
6. Import FinRL submodule in files.
We need to make sure the python interpreter detects FinRL as a module:

- If you are using Visual Studio Code: use cmd+shift+P to open the shortcut in VS Code, type "Preferences: Open Workspace Settings (JSON)". Add this into `settings.json`:
```json
{
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "/path/to/module/FinRL"
    }
}
```
- It equals to `export PYTHONPATH=/some/path/` every time we opened the terminal
- The good thing of using submodule is that we can set break point within the submodule and debug from an outside file which imports it

## Notes: Edit within the submodule
A submodule is essentially a separate Git repository embedded within another Git repository (the "superproject"). You can treat the submodule directory as a regular cloned repository and perform all standard Git operations within it.
Here's how you would typically work with edits in a submodule: navigate into the submodule directory.
Code

    cd path/to/your/submodule
Make your desired changes: Edit files, add new ones, etc., just as you would in any other Git repository.
Commit the changes within the submodule:
Code

    git commit -am "Descriptive commit message for submodule changes"
Push the submodule changes to its remote repository (if applicable):
Code

    git push origin your_branch_name
It is crucial to push submodule changes to their own remote repository before updating the superproject to reference them. Navigate back to the superproject.
Code

    cd ../.. # Or wherever your superproject root is
Commit the updated submodule reference in the superproject: Git will detect that the submodule's commit reference has changed.
Code

    git add path/to/your/submodule
    git commit -m "Update submodule to new commit"
Push the superproject changes.
Code

    git push origin main # Or your superproject's branch