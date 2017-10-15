# -*- encoding: ascii -*-
"""
Checking tasks
~~~~~~~~~~~~~~

"""

import invoke as _invoke

from . import clean as _clean


@_invoke.task(_clean.py, default=True)
def lint(ctx):
    """ Run pylint """
    pylint = ctx.shell.frompath('pylint')
    if pylint is None:
        raise RuntimeError("pylint not found")

    with ctx.shell.root_dir():
        ctx.run(ctx.c(
            r''' %(pylint)s --rcfile pylintrc %(package)s ''',
            pylint=pylint,
            package=ctx.package
        ), echo=True)
