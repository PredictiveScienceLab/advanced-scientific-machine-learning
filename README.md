# ME 697 - Advanced Scientific Machine Learning

This is the GitHub repository for the experimental course ME 697 - Advanced Scientific Machine Learning at Purdue University.
The course is currently being developed.
The material is very volatile.
Please check the tagged `sp2024` version if you want something stable.

## Production phase details for TAs creating the material
+ I suggest working with VS Code
+ Install the Python extension
+ Install the Jupyter extension
+ Install `jupyter-book` by running
```bash
pip install jupyter-book
```
+ Start your Jupyter notebooks with
```python
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
import seaborn as sns
sns.set_context("paper")
sns.set_style("ticks");
```
+ Use tags to hide input or output of cells that are not relevant for the students. The tags are `hide-input` and `hide-output`.
+ Work on a local copy of the repository and push to the remote repository when you are done.
+ Compile the local copy by:
```bash
jupyter-book build book
```
+ Check the local copy by opening the file `_build/html/index.html` in a browser.