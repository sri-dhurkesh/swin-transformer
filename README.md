# Swin Transformer

This repository contains the implementation of the Swin Transformer model, a hierarchical vision transformer using shifted windows. The project is structured to facilitate training, evaluation, and experimentation with the Swin Transformer model.

## Repository Structure

- `assets/`: Contains assets related to the project.
- `configs/`: Configuration files for different versions of the Swin Transformer.
- `data/`: Contains datasets used for training and evaluation.
- `models/`: Contains the implementation of the Swin Transformer model and its components.
  - `swin_v1/`: Implementation of the first version of the Swin Transformer.
    - `patch_embedding.py`: Code for patch embedding.
    - `patch_merging.py`: Code for patch merging.
    - `swin_transformer.py`: Main implementation of the Swin Transformer model.
    - `swin_transformer_block/`: Contains the implementation of the Swin Transformer blocks.
- `notebooks/`: Jupyter notebooks for experimentation and visualization.
- `main.py`: Entry point for training and evaluating the model.
- `pyproject.toml`: Project configuration file for Poetry.
- `poetry.lock`: Lock file for Poetry dependencies.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/swin-transformer.git
   cd swin-transformer
   ```
2. Install the dependencies using Poetry:

   ```sh
   poetry install
   ```

3. Run the main script to train or evaluate the model:
   ```sh
   python main.py
   ```

### Customizations

The repository includes a `.devcontainer` configuration for Visual Studio Code, which sets up a development environment with the necessary extensions and tools.
