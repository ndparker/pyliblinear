# -*- coding: ascii -*-
u"""
:Copyright:

 Copyright 2015
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

===============================
 Tests for pyliblinear.Problem
===============================

Tests for pyliblinear.Problem
"""
__author__ = u"Andr\xe9 Malo"
__docformat__ = "restructuredtext en"

from nose.tools import (
    assert_equals, assert_raises
)

import pyliblinear as _pyliblinear


# pylint: disable = protected-access


def test_problem_from_iterable_dict():
    """ Problem.from_iterable from dicts """
    problem = _pyliblinear.Problem.from_iterable(
        [(1, {3: 4, 1: 7}), (2, {2: 1})],
    )

    assert_equals(problem.bias(), None)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [1.0, 2.0])
    assert_equals(list(problem.features()), [{1: 7.0, 3: 4.0}, {2: 1.0}])


def test_problem_from_iterable_dict_assign():
    """ Problem.from_iterable from dicts with assigned labels """
    problem = _pyliblinear.Problem.from_iterable(
        [{3: 4, 1: 7}, {2: 1}],
        assign_labels=4,
    )

    assert_equals(problem.bias(), None)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [4.0, 4.0])
    assert_equals(list(problem.features()), [{1: 7.0, 3: 4.0}, {2: 1.0}])


def test_problem_from_iterable_dict_bias():
    """ Problem.from_iterable from dicts with bias """
    problem = _pyliblinear.Problem.from_iterable(
        [(1, {3: 4, 1: 7}), (2, {2: 1})],
        bias=2.0,
    )

    assert_equals(problem.bias(), 2.0)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [1.0, 2.0])
    assert_equals(list(problem.features()), [{1: 7.0, 3: 4.0}, {2: 1.0}])


def test_problem_from_iterable_dict_bias_assign():
    """ Problem.from_iterable from dicts with bias and assign_labels """
    problem = _pyliblinear.Problem.from_iterable(
        [{3: 4, 1: 7}, {2: 1}],
        bias=3.0,
        assign_labels=5,
    )

    assert_equals(problem.bias(), 3.0)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [5.0, 5.0])
    assert_equals(list(problem.features()), [{1: 7.0, 3: 4.0}, {2: 1.0}])


def test_problem_from_iterable_tuple():
    """ Problem.from_iterable from tuples """
    problem = _pyliblinear.Problem.from_iterable(
        [(1, (4, 6)), (2, (1,))],
    )

    assert_equals(problem.bias(), None)
    assert_equals(problem.width(), 2)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [1.0, 2.0])
    assert_equals(list(problem.features()), [{1: 4.0, 2: 6.0}, {1: 1.0}])


def test_problem_from_iterable_tuple_assign():
    """ Problem.from_iterable from tuples with assigned labels """
    problem = _pyliblinear.Problem.from_iterable(
        [(5, 8), (1, 3, 6)],
        assign_labels=4,
    )

    assert_equals(problem.bias(), None)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [4.0, 4.0])
    assert_equals(list(problem.features()), [
        {1: 5.0, 2: 8.0}, {1: 1.0, 2: 3.0, 3: 6.0}
    ])


def test_problem_from_iterable_tuple_bias():
    """ Problem.from_iterable from tuples with bias """
    problem = _pyliblinear.Problem.from_iterable(
        [(1, (4, 6)), (2, (1,))],
        bias=5.0,
    )

    assert_equals(problem.bias(), 5.0)
    assert_equals(problem.width(), 2)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [1.0, 2.0])
    assert_equals(list(problem.features()), [{1: 4.0, 2: 6.0}, {1: 1.0}])


def test_problem_from_iterable_tuple_bias_assign():
    """ Problem.from_iterable from tuples with bias and assign_labels """
    problem = _pyliblinear.Problem.from_iterable(
        [(5, 8), (1, 3, 6)],
        bias=3.0,
        assign_labels=7,
    )

    assert_equals(problem.bias(), 3.0)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [7.0, 7.0])
    assert_equals(list(problem.features()), [
        {1: 5.0, 2: 8.0}, {1: 1.0, 2: 3.0, 3: 6.0}
    ])


class KeyIterator(object):
    """ Sequence wrapper providing keys method """

    def __init__(self, vector):
        self._vector = vector

    def iterkeys(self):
        """ Return key iterator """
        return xrange(1, len(self._vector) + 1)

    def __getitem__(self, idx):
        return self._vector[idx - 1]


def test_problem_from_iterable_keys():
    """ Problem.from_iterable from key iterator """
    problem = _pyliblinear.Problem.from_iterable(
        [(3, KeyIterator((5, 8))), (4, KeyIterator((1, 3, 6)))]
    )

    assert_equals(problem.bias(), None)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [3.0, 4.0])
    assert_equals(list(problem.features()), [
        {1: 5.0, 2: 8.0}, {1: 1.0, 2: 3.0, 3: 6.0}
    ])


def test_problem_from_iterable_keys_assign():
    """ Problem.from_iterable from key iterator and assigned labels """
    problem = _pyliblinear.Problem.from_iterable(
        [KeyIterator((5, 8)), KeyIterator((1, 3, 6))],
        assign_labels=2.0,
    )

    assert_equals(problem.bias(), None)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [2.0, 2.0])
    assert_equals(list(problem.features()), [
        {1: 5.0, 2: 8.0}, {1: 1.0, 2: 3.0, 3: 6.0}
    ])


def test_problem_from_iterable_keys_bias():
    """ Problem.from_iterable from key iterator with bias"""
    problem = _pyliblinear.Problem.from_iterable(
        [(3, KeyIterator((5, 8))), (4, KeyIterator((1, 3, 6)))],
        bias=9.0,
    )

    assert_equals(problem.bias(), 9.0)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [3.0, 4.0])
    assert_equals(list(problem.features()), [
        {1: 5.0, 2: 8.0}, {1: 1.0, 2: 3.0, 3: 6.0}
    ])


def test_problem_from_iterable_keys_bias_assign():
    """
    Problem.from_iterable from key iterator with bias and assigned labels
    """
    problem = _pyliblinear.Problem.from_iterable(
        [KeyIterator((5, 8)), KeyIterator((1, 3, 6))],
        assign_labels=4.0,
        bias=1.0,
    )

    assert_equals(problem.bias(), 1.0)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [4.0, 4.0])
    assert_equals(list(problem.features()), [
        {1: 5.0, 2: 8.0}, {1: 1.0, 2: 3.0, 3: 6.0}
    ])


def test_problem_from_iterables_dict():
    """ Problem.from_iterables from dicts """
    problem = _pyliblinear.Problem.from_iterables(
        [2, 3], [{3: 4, 1: 7}, {2: 1}],
    )

    assert_equals(problem.bias(), None)
    assert_equals(problem.width(), 3)
    assert_equals(problem.height(), 2)
    assert_equals(list(problem.labels()), [2.0, 3.0])
    assert_equals(list(problem.features()), [{1: 7.0, 3: 4.0}, {2: 1.0}])


def test_problem_from_iterables_exc():
    """ Problem.from_iterables raises exception on different lengths """
    with assert_raises(ValueError):
        _pyliblinear.Problem.from_iterables(
            [2, 3], [{3: 4, 1: 7}, {2: 1}, {2, 1}],
        )