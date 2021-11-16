
import sys,os,string,glob,subprocess

from setuptools import setup,Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

long_description = """\
darkCNN
"""


version='0.0.1'
         
    
INCDIRS=['.']

packages = ['darkCNN']
package_dir = {'darkCNN':'./src'}





setup   (       name            = "darkCNN",
                version         = version,
                author          = "David Harvey",
                author_email    = "davidharvey1986@googlemail.com",
                description     = "darkCNN module",
                license         = 'MIT',
                packages        = packages,
                package_dir     = package_dir,
                scripts         = ['bin/darkCNN'],
                url = 'https://github.com/davidharvey1986/darkCNN', # use the URL to the github repo
                download_url = 'https://github.com/davidharvey1986/darkCNN/archive/'+version+'.tar.gz',
                install_requires=['tensorFlow', 'astropy==4.0', 'matplotlib==3.3.3','PyQt5','keras'],
        )
