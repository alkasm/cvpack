from setuptools import setup, find_packages

setup(
    name="cvtools",
    author="Alexander Reynolds <alkasm>",
    author_email="ar@reynoldsalexander.com",
    url="https://github.com/alkasm/cvtools",
    version="0.4",
    packages=find_packages(),
    install_requires=["opencv-python"],
    license="MIT",
    long_description=open("README.md").read(),
)
