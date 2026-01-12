# FedSSR - Federated Learning Experiment Framework

FedSSR is a Python framework for federated learning experiments, supporting research on various attack and defense methods.

## Project Structure

### Core Files

#### `src/main.py`
- **Purpose**: Main entry point of the project
- **Features**: 
  - Provides command-line interface using Typer
  - Loads TOML configuration files
  - Supports command-line parameter overrides for config file parameters
  - Sets up logging system
  - Launches federated learning experiments

#### `batch_runner.py`
- **Purpose**: Batch experiment runner
- **Features**:
  - Supports running single configuration file or batch running all configs
  - Manages multiple parallel experiments using tmux sessions
  - Automatically cleans old log files
  - Supports all command-line parameters of main.py

#### `batch.sh`
- **Purpose**: Shell script version of batch runner
- **Features**:
  - Traverses all TOML configuration files in `conf/` directory
  - Creates independent tmux sessions for each configuration
  - Supports `--method` parameter override
  - Automatically cleans old logs and sessions

#### `process_logs.py`
- **Purpose**: Experiment result processing and analysis tool
- **Features**:
  - Reads experiment logs in Parquet format
  - Computes mean and standard deviation of Top-5 accuracy
  - Supports multiple output formats (LaTeX, Markdown, CSV, Excel)
  - Generates experiment result comparison tables

#### `logger.py`
- **Purpose**: Logging system configuration tool
- **Features**: Displays loguru logging level information

### Configuration Files

#### `conf/` Directory
- **Purpose**: Stores experiment configuration files

#### Configuration Parameters
- **Task Parameters**: Model architecture, dataset, data splitting strategy
- **System Parameters**: Number of clients, number of servers, training rounds
- **Training Parameters**: Learning rate, batch size, device configuration
- **Security Parameters**: Attack type, aggregation method, selection ratio

### Log Files

#### `log/` Directory
- **Purpose**: Stores experiment logs and results
- **Structure**:
  - `*.log`: Experiment logs in text format
  - `*.parquet`: Structured experiment data
  - Various subdirectories store results for different experiment types

### Project Configuration

#### `pyproject.toml`
- **Purpose**: Python project configuration file
- **Features**:
  - Defines project dependencies (PyTorch, NumPy, Pandas, etc.)
  - Configures UV package manager
  - Sets PyTorch index source (CPU/CUDA)

#### `.python-version`
- **Purpose**: Specifies Python version (3.13)

#### `uv.lock`
- **Purpose**: Lock file for UV package manager, ensuring consistent dependency versions

## Usage

### Running Single Experiment
```bash
uv run -m src.main -c conf/0.toml
```

### Running Batch Experiments
```bash
# Using Python batch runner
uv run batch_runner.py --batch

# Using Shell script
./batch.sh --method ours
```

### Processing Results
```bash
uv run process_logs.py --path "log/resnet/**/*.parquet" --format markdown
```

## Key Features

- Supports multiple federated learning attack methods (Ascent, Lie, MinMax, etc.)
- Flexible configuration system with command-line parameter override support
- Parallel experiment execution using tmux session management
- Comprehensive logging and result analysis tools
- Support for multiple deep learning models and datasets
- Extensible experimental framework design

## Dependencies

- Python 3.13+
- PyTorch 2.7.0+
- UV package manager
- tmux (for parallel experiment management)