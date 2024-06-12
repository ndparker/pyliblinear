# -*- coding: ascii -*-
r"""
:Copyright:

 Copyright 2015 - 2024
 Andr\xe9 Malo or his licensors, as applicable

:License:

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================
 Tests for pyliblinear.Solver
==============================

Tests for pyliblinear.Solver.
"""
__author__ = u"Andr\xe9 Malo"

import pyliblinear as _pyliblinear


def test_solver_default():
    """Solver initializes with default arguments"""
    solver = _pyliblinear.Solver()

    assert solver.type == 'L2R_L2LOSS_SVC_DUAL'
    assert solver.C == 1.0
    assert solver.eps == 0.1
    assert solver.p == 0.1
    assert solver.weights() == {}


def test_solver_types():
    """Solver accepts different solver types"""
    tests = [
        'L2R_LR',
        'L2R_L2LOSS_SVC',
        'L2R_L2LOSS_SVR',
        'L2R_L2LOSS_SVC_DUAL',
        'L2R_L1LOSS_SVC_DUAL',
        'MCSVM_CS',
        'L2R_LR_DUAL',
        'L1R_L2LOSS_SVC',
        'L1R_LR',
        'L2R_L2LOSS_SVR_DUAL',
        'L2R_L1LOSS_SVR_DUAL',
    ]
    for stype in tests:
        solver = _pyliblinear.Solver(_pyliblinear.SOLVER_TYPES[stype])
        assert solver.type == stype

        solver = _pyliblinear.Solver(stype)
        assert solver.type == stype


def test_solver_eps_defaults():
    """Solver initializes eps depending on the solver type"""
    tests = [
        ('L2R_LR', 0.01),
        ('L2R_L2LOSS_SVC', 0.01),
        ('L2R_L2LOSS_SVR', 0.001),
        ('L2R_L2LOSS_SVC_DUAL', 0.1),
        ('L2R_L1LOSS_SVC_DUAL', 0.1),
        ('MCSVM_CS', 0.1),
        ('L2R_LR_DUAL', 0.1),
        ('L1R_L2LOSS_SVC', 0.01),
        ('L1R_LR', 0.01),
        ('L2R_L2LOSS_SVR_DUAL', 0.1),
        ('L2R_L1LOSS_SVR_DUAL', 0.1),
    ]
    for solver_type, eps in tests:
        solver = _pyliblinear.Solver(solver_type)

        assert solver.type == solver_type
        assert solver.C == 1.0
        assert solver.eps == eps
        assert solver.p == 0.1
        assert solver.weights() == {}


def test_solver_param():
    """Solver accepts different parameters"""
    solver = _pyliblinear.Solver(
        'L1R_LR', C=0.25, eps=0.0001, p=3, weights={2: 5, 3: 4, 6: 9.5}
    )

    assert solver.type == 'L1R_LR'
    assert solver.C == 0.25
    assert solver.eps == 0.0001
    assert solver.p == 3.0
    assert solver.weights() == {2: 5.0, 3: 4.0, 6: 9.5}
