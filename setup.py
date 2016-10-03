from distutils.core import setup

# Read the version number
with open("mtm_stats/_version.py") as f:
    exec(f.read())

setup(
    name='mtm_stats',
    version=__version__, # use the same version that's in _version.py
    author='David N. Mashburn',
    author_email='david.n.mashburn@gmail.com',
    packages=['mtm_stats'],
    scripts=[],
    license='LICENSE.txt',
    description='Highly efficient set statistics about many-to-many relationships',
    long_description=open('README.txt').read(),
    install_requires=[
                      
                     ],
)
