from setuptools import find_packages, setup

from megbox import __author__, __version__

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
