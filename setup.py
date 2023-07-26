from setuptools import setup, find_packages

setup(
    name='segmented_linear_model_selection',
    version='0.0.99',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'statsmodels',
        'matplotlib'
    ],
    author='Scott Tyler',
    description='A package for fitting and comparing segmented linear models based on given intervals.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)

