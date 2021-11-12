from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
requirements = []
    
setup(
    name='getspan',
    version='1.0.0',
    author='Christine Dien',
    description='To identify gene spans along a pseudo-axis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/settylab/getspan',
    packages=['getspan'],
    install_requires=requirements,
)