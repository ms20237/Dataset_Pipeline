# Fiftyone dataset Pipeline


## Introduction
This repository implements the **FiftyTone dataset pipeline**, a modular data processing workflow to prepare, transform, and validate datasets for downstream tasks (e.g. training, evaluation, analysis).

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Installation & Dependencies](#installation--dependencies)  
- [Usage](#usage)  
- [Configuration](#configuration)  
- [Examples](#examples)  
- [Development & Testing](#development--testing)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

The **FiftyTone Pipeline** is designed to manage the lifecycle of dataset preparation:
1. Ingest raw data (images, annotations, etc.)  
2. Apply preprocessing / transformation  
3. Validate / clean data  
4. Output in formats consumable by model training or analysis  

This pipeline helps to standardize dataset creation, reduce errors, and maintain reproducibility.

---

## Features

- Configurable via parameter files  
- Support for multiple input / output formats  

---

## Repository Structure

Here’s a sample layout of the pipeline directory:
```bash
fiftytone_pipeline/
├── configs/ # Config files (YAML / JSON)
├── shuttrix/ # tasks and operators and main pipeline file
    ├── operators/ visualizer and field generators classes
    ├── pipelines/ dataset and model pipeline
    ├── sandbox/ dont care scripts :)
    └── tasks/ dataset and model scipts 
└── sim_models/  name of model for finding similar samples in dataset
```
