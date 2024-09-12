#!/bin/bash
# Usage: bash experiment.sh <project>
# Description: Run the experiment for the given project

project=$1
python gen_semantic.py $project &&python run.py $project 0 && python run.py $project 1 && python run.py $project 2 && python run.py $project 4