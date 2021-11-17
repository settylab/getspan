from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name='getspan',
    version='1.0.0',
    author='Christine Dien',
    description='To identify gene spans along a pseudo-axis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/settylab/getspan',
    package_dir={"": "src"},
    packages=['getspan'],
    install_requires=[
        "pandas>=1.3.2",
        "numpy>=1.20.3",
        "scanpy>=1.7.2",
        "sklearn",
        "tqdm>=4.32.2",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1"
    ],
     python_requires=">=3.6"
)
