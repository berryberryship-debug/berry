"""
__main__.py
-----------
Allows the package to be run directly::

    python -m nonlinear_information_system
    python -m nonlinear_information_system --config path/to/config.yaml
"""

from nonlinear_information_system.cli import main

if __name__ == "__main__":
    main()
