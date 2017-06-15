from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='pylinea',
    version='0.1.0',
    description='Linear algebra functions',
    long_description=readme,
    author='K. Isom',
    author_email='kyle@imap.cc',
    url='https://git.metacircular.net/kyle/pylinea',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

