#!/usr/bin/env python
# -*- coding: ascii -*-
#
# Copyright 2015
# Andr\xe9 Malo or his licensors, as applicable
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from _setup import run


def setup(args=None, _manifest=0):
    """ Main setup function """
    from _setup.ext import Extension

    return run(
        script_args=args,
        manifest_only=_manifest,
        ext=[
            Extension('pyliblinear._liblinear', [
                "pyliblinear/bufwriter.c",
                "pyliblinear/compat.c",
                "pyliblinear/iter.c",
                "pyliblinear/main.c",
                "pyliblinear/matrix.c",
                "pyliblinear/model.c",
                "pyliblinear/solver.c",
                "pyliblinear/tokreader.c",
                "pyliblinear/util.c",
                "pyliblinear/vector.c",

                "pyliblinear/liblinear/blas/ddot.c",
                "pyliblinear/liblinear/blas/dscal.c",
                "pyliblinear/liblinear/blas/dnrm2.c",
                "pyliblinear/liblinear/blas/daxpy.c",
                "pyliblinear/liblinear/linear.cpp",
                "pyliblinear/liblinear/tron.cpp",
            ], depends=[
                "pyliblinear/pyliblinear.h",

                "pyliblinear/liblinear/linear.h",
                "pyliblinear/liblinear/tron.h",
                "pyliblinear/liblinear/blas/blasp.h",
                "pyliblinear/liblinear/blas/blas.h",
            ], include_dirs=[
                "pyliblinear",
                "pyliblinear/liblinear",
                "pyliblinear/liblinear/blas",
            ])
        ],
    )


def manifest():
    """ Create List of packaged files """
    return setup((), _manifest=1)


if __name__ == '__main__':
    setup()
