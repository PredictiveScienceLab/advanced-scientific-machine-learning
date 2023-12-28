# This script makes and publishes the lecture book
# Author:
# 	Ilias Bilionis
# Date:
# 	5/9/2022
#   12/28/2023

# Make it
jupyter-book build book --all

# Publish it
ghp-import -n -p -f book/_build/html