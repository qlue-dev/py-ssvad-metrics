
import os
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(here, 'ssvad_metrics/VERSION'), encoding='utf-8') as f:
    version = f.read()

setup(
    name='py-ssvad_metrics',
    version=version,
    description='Single Scene Video Anomaly Detection Metrics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/qlue-dev/py-ssvad-metrics',
    author='PT Qlue Performa Indonesia',
    author_email='developer@qlue.id',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Development Status :: 4 - Beta'
    ],
    packages=find_packages(),
    package_data={
        "": ["VERSION"],
    },
    python_requires='>=3.6,<4',
    install_requires=[
        'numpy>=1.18',
        'typing_extensions>=3.7.4',
        'pytest>=6.2.3',
        'scikit-learn>=0.24.1',
        'scipy>=1.5.4',
        'pydantic>=1.8.1',
        'pandas>=1.1.5']
)
