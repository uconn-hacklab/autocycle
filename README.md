# autocycle

Autocycle is hacklab's first major project to develop an autonomous riderless bike.

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/uconn-hacklab/autocycle.git
   cd autocycle
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the project and its dependencies:
   ```
   pip install -e .
   ```
   This command uses the `pyproject.toml` file to install all necessary dependencies.

### Development Setup

For development, you can install additional tools by including the `dev` extra:

```
pip install -e .[dev]
```

This will install development tools specified in the `pyproject.toml` file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.