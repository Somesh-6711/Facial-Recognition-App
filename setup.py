from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> list[str]:
    '''
    this function returns a list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name='Facial Recognition APP',
    version='0.0.1',
    author='Somesh',
    author_email='100meshp@gmail.com',
    packages=find_packages(),
    url='https://github.com/Somesh-6711/Facial-Recognition-App.git',
    install_requires=get_requirements('requirements.txt'),
)