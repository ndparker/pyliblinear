# -*- encoding: ascii -*-
"""
Cleanup tasks
~~~~~~~~~~~~~

"""

import invoke as _invoke


@_invoke.task()
def py(ctx):
    """ Wipe *.py[co] files """
    for name in ctx.shell.files('.', '*.py[co]'):
        ctx.shell.rm(name)
    for name in ctx.shell.dirs('.', '__pycache__'):
        ctx.shell.rm_rf(name)


@_invoke.task(py)
def dist(ctx):
    """ Wipe all """
    clean(ctx, so=True, cache=True)


@_invoke.task(py, default=True)
def clean(ctx, so=False, cache=False):
    """ Wipe *.py[co] files and test leftovers """
    for name in ctx.shell.files('.', '.coverage*', recursive=False):
        ctx.shell.rm(name)
    ctx.shell.rm('gcov.out')
    ctx.shell.rm_rf(
        'docs/coverage',
        'docs/gcov',
        'build',
        'dist',
        ctx.doc.userdoc,
        ctx.doc.website.source,
        ctx.doc.website.target,
    )
    if cache:
        cacheclean(ctx)
    if so:
        soclean(ctx)


@_invoke.task()
def cacheclean(ctx):
    """ Wipe Cache files """
    ctx.shell.rm_rf(
        '.tox',
        '.cache',
        'tests/.cache',
    )


@_invoke.task()
def soclean(ctx):
    """ Wipe *.so files """
    for name in ctx.shell.files('.', '*.pyd'):
        ctx.shell.rm(name)
    for name in ctx.shell.files('.', '*.so'):
        ctx.shell.rm(name)
