import os

def is_package(path):
    return (
        os.path.isdir(path) and
        os.path.isfile(os.path.join(path, '__init__.py'))
        )

def find_packages(path, base="" ):
    """ Find all packages in path """
    packages = {}
    for item in os.listdir(path):
        dir = os.path.join(path, item)
        if is_package( dir ):
            if base:
                module_name = "%(base)s.%(item)s" % vars()
            else:
                module_name = item
            packages[module_name] = dir
            packages.update(find_packages(dir, module_name))
    return packages

def find_package_data(package,data):
    """Find the package data"""

    package_data ={} 
    for pack in packages.keys():
            package_data[pack] = data 
    return package_data

def write_manifest(package,data):
    """Find the package data"""
    filename ='MANIFEST.in'
    FILE = open(filename,"w")

    for item in data:
        for pack in packages.keys():
            line = "include %s/%s\n" % (pack, item) 
            FILE.writelines(line)
    FILE.close()

packages = find_packages(".")
data_format = ['*.neu','*.mat']
write_manifest(packages,data_format)
#package_data = find_package_data(packages,data_format)

print packages
from distutils.core import setup
setup(name='pynudg',
      version='1.0',
      package_dir = packages,
      packages = packages.keys(), 
    )

