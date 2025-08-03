# Installation

## Requirements

HPI-FHFA requires Python 3.8 or higher. The system has been tested on:
- Python 3.8, 3.9, 3.10, 3.11, 3.12
- Operating Systems: Linux, macOS, Windows

## Quick Install

### Using pip

```bash
pip install hpi-fhfa
```

### From Source

```bash
git clone https://github.com/yourusername/hpi-fhfa.git
cd hpi-fhfa
pip install -e .
```

## Development Installation

For development work, install with additional dependencies:

```bash
git clone https://github.com/yourusername/hpi-fhfa.git
cd hpi-fhfa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

## Dependencies

### Core Dependencies
- **numpy** >= 1.20.0: Numerical computing
- **pandas** >= 1.3.0: Data manipulation
- **scipy** >= 1.7.0: Scientific computing
- **scikit-learn** >= 0.24.0: Machine learning utilities
- **geopandas** >= 0.10.0: Geographic data handling
- **shapely** >= 1.8.0: Geometric operations

### API Dependencies
- **flask** >= 2.0.0: Web framework
- **flask-cors** >= 3.0.0: CORS support
- **gunicorn** >= 20.0.0: Production server

### Optional Dependencies
- **matplotlib** >= 3.4.0: Visualization
- **seaborn** >= 0.11.0: Statistical graphics
- **plotly** >= 5.0.0: Interactive visualizations

### Development Dependencies
- **pytest** >= 7.0.0: Testing framework
- **pytest-cov** >= 3.0.0: Coverage reporting
- **black** >= 22.0.0: Code formatting
- **flake8** >= 4.0.0: Linting
- **mypy** >= 0.900: Type checking
- **sphinx** >= 4.0.0: Documentation

## Docker Installation

### Using Pre-built Image

```bash
docker pull hpi-fhfa:latest
docker run -p 8000:8000 hpi-fhfa:latest
```

### Building from Dockerfile

```bash
git clone https://github.com/yourusername/hpi-fhfa.git
cd hpi-fhfa
docker build -t hpi-fhfa .
docker run -p 8000:8000 hpi-fhfa
```

### Docker Compose

For a complete setup with database and cache:

```bash
docker-compose up -d
```

## Verification

Verify your installation:

```python
import hpi_fhfa
print(hpi_fhfa.__version__)

# Run basic test
from hpi_fhfa import HPICalculator
calculator = HPICalculator()
print("Installation successful!")
```

## Platform-Specific Notes

### macOS
- Requires Xcode Command Line Tools for some dependencies
- Install with: `xcode-select --install`

### Windows
- Some geographic libraries may require additional setup
- Consider using WSL2 for best compatibility

### Linux
- May need system libraries for geographic operations:
  ```bash
  sudo apt-get install libgeos-dev libproj-dev
  ```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'hpi_fhfa'
   ```
   Solution: Ensure you've activated your virtual environment

2. **Dependency Conflicts**
   ```
   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
   ```
   Solution: Create a fresh virtual environment

3. **Geographic Library Issues**
   ```
   OSError: Could not find library geos_c or load any of its variants
   ```
   Solution: Install system dependencies (see platform-specific notes)

### Getting Help

If you encounter issues:
1. Check the [FAQ](faq.md)
2. Search existing [GitHub Issues](https://github.com/yourusername/hpi-fhfa/issues)
3. Create a new issue with:
   - Python version
   - Operating system
   - Full error traceback
   - Steps to reproduce

## Next Steps

- Continue to [Quick Start](quickstart.md) to run your first calculation
- See [Configuration](user_guide/configuration.md) for customization options
- Review [API Reference](api_reference/index.md) for detailed documentation