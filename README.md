# pyliblinear - a liblinear python API

TABLE OF CONTENTS
-----------------

1. Introduction
1. Development Status
1. Copyright and License
1. System Requirements
1. Installation
1. Documentation
1. Bugs
1. Author Information


## INTRODUCTION

**pyliblinear** is an API for
[liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) without using ctypes.
It aims for more pythonic access to liblinear's features, especially featuring
streams of data and lazy evaluations whereever possible.

* [Change Log](docs/CHANGES)
* [Development](docs/DEVELOPMENT.md)


DEVELOPMENT STATUS
------------------

Alpha.
The API is not stabilized yet. There also may be a few kinks here and there.


## COPYRIGHT AND LICENSE

Copyright 2015 - 2017
André Malo or his licensors, as applicable.

The whole package is distributed under the Apache License Version 2.0.
You'll find a copy in the root directory of the distribution or online
at: <http://www.apache.org/licenses/LICENSE-2.0>.

pyliblinear ships with a copy of
[liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), which is licensed
under the terms of the [3-clause BSD
license](http://opensource.org/licenses/BSD-3-Clause).


## SYSTEM REQUIREMENTS

You need at least python 2.7 or Python 3 starting with version 3.4.

You also need a build environment for python C/C++ extensions (i.e. a compiler
and the python development files).


## INSTALLATION

### Using pip

```
$ pip install pyliblinear
```


### Using distutils

Download the package, unpack it, change into the directory

```
$ python setup.py install
```

The command above will install a new "pyliblinear" package into python's
library path.


## DOCUMENTATION

You'll find a user documentation in the `docs/userdoc/` directory of the
distribution package.

The latest documentation is also available online at
<http://opensource.perlig.de/pyliblinear/>.


## BUGS

No bugs, of course. ;-)

But if you've found one or have an idea how to improve pyliblinear, feel free to
send a pull request on [github](https://github.com/ndparker/pyliblinear) or
send a mail to <pyliblinear-bugs@perlig.de>.


## AUTHOR INFORMATION

André "nd" Malo <nd@perlig.de>, GPG: 0x8103A37E


>  If God intended people to be naked, they would be born that way.
>                                                   -- Oscar Wilde
