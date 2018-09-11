from setuptools import find_packages
from setuptools import setup
import sys, os, os.path

REQUIRED_PACKAGES = [
  'google-cloud-storage',
  'tensorflow-gpu>=1.8',
  'lmdb',
  'opencv-python',
  'matplotlib',
  'Pillow',
]

setup(
    name='jonasz_master_thesis',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)
