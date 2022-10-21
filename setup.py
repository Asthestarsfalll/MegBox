from setuptools import setup, find_packages
from megbox import __version__, __author__

if __name__ == "__main__":
    setup(
        name='megbox',
        version=__version__,
        author=__author__,
        description="pass",
        license='MIT',
        packages=find_packages(),
        python_requires='>=3.6',
    )
