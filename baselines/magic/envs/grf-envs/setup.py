from setuptools import setup, find_packages

setup(name='grf_envs',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym','numpy', 'gfootball']  
)  