# netneurotools
This toolbox is a collection of functions written in Python and Matlab that get frequent usage in the [Network Neuroscience Lab](www.misiclab.com), housed in the [Brain Imaging Centre](https://www.mcgill.ca/bic/home) at [McGill University](http://www.mcgill.ca/).

## Requirements
#### Python
The Python tools require Python >= 3.5 (and will yell at you if you try and install it on any earlier versions).
#### Matlab
Matlab tools should work on R2012B or later (though that is an estimate based on the versions of Matlab that were around when the code was written).

## Installation
#### Python
You can use `python setup.py install` to install the Python tools on your system. Access the tools via `import netneurotools`.
#### Matlab
Simply add the `netneuro_matlab` directory to your Matlab path in order to access those functions.

## Usage

In general, all the Python and Matlab tools should have similar names and function calls. Many of the Python functions are translations from original Matlab code, so they may be only marginally Pythonic.

## Bugs and Questions

If you find any bugs, let us know by raising an [issue](https://github.com/netneurolab/netneurotools/issues) and one of the lab members will try and get it fixed quickly. Better yet, submit a [pull request](https://github.com/netneurolab/netneurotools/pulls)!

There isn't any official documentation for now, but most of the functions have relatively comprehensive docstrings. If you can't figure something out, raise an issue and we'll do what we can!

## License

All tools are released under the MIT license; the full details of the license can be found in the `LICENSE` file.
