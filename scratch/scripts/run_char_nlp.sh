#!/bin/bash

# Set the environment variable to suppress OpenMP warnings
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the script using pixi
pixi run python src/scratch/char_level_nlp.py
