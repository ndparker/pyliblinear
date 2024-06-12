# -*- coding: ascii -*-
u"""
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

=====================================
 Tests for pyliblinear.FeatureMatrix
=====================================

Tests for pyliblinear.FeatureMatrix
"""
__author__ = u"Andr\xe9 Malo"

import os as _os

from pytest import raises

import pyliblinear as _pyliblinear


# pylint: disable = protected-access


def test_matrix_dict():
    """FeatureMatrix from dicts"""
    matrix = _pyliblinear.FeatureMatrix(
        [(1, {3: 4, 1: 7}), (2, {2: 1})],
    )

    assert matrix.width == 3
    assert matrix.height == 2
    assert list(matrix.labels()) == [1.0, 2.0]
    assert list(matrix.features()) == [{1: 7.0, 3: 4.0}, {2: 1.0}]


def test_matrix_load_save(tmpdir):
    """FeatureMatrix load / save"""
    matrix = _pyliblinear.FeatureMatrix(
        [(1, {3: 4, 1: 7}), (2, {2: 1})],
    )
    filename = _os.path.join(str(tmpdir), 'matrix_load_save.matrix')
    matrix.save(filename)
    matrix = _pyliblinear.FeatureMatrix.load(filename)

    assert matrix.width == 3
    assert matrix.height == 2
    assert list(matrix.labels()) == [1.0, 2.0]
    assert list(matrix.features()) == [{1: 7.0, 3: 4.0}, {2: 1.0}]


def test_matrix_dict_assign():
    """FeatureMatrix from dicts with assigned labels"""
    matrix = _pyliblinear.FeatureMatrix(
        [{3: 4, 1: 7}, {2: 1}],
        assign_labels=4,
    )

    assert matrix.width == 3
    assert matrix.height == 2
    assert list(matrix.labels()) == [4.0, 4.0]
    assert list(matrix.features()) == [{1: 7.0, 3: 4.0}, {2: 1.0}]


def test_matrix_tuple():
    """FeatureMatrix from tuples"""
    matrix = _pyliblinear.FeatureMatrix(
        [(1, (4, 6)), (2, (1,))],
    )

    assert matrix.width == 2
    assert matrix.height == 2
    assert list(matrix.labels()) == [1.0, 2.0]
    assert list(matrix.features()) == [{1: 4.0, 2: 6.0}, {1: 1.0}]


def test_matrix_tuple_assign():
    """FeatureMatrix from tuples with assigned labels"""
    matrix = _pyliblinear.FeatureMatrix(
        [(5, 8), (1, 3, 6)],
        assign_labels=4,
    )

    assert matrix.width == 3
    assert matrix.height == 2
    assert list(matrix.labels()) == [4.0, 4.0]
    assert list(matrix.features()) == [
        {1: 5.0, 2: 8.0},
        {1: 1.0, 2: 3.0, 3: 6.0},
    ]


try:
    xrange
except NameError:
    xrange = range  # pylint: disable = redefined-builtin


class KeyIterator(object):
    """Sequence wrapper providing keys method"""

    def __init__(self, vector):
        self._vector = vector

    def iterkeys(self):
        """Return key iterator"""
        return xrange(1, len(self._vector) + 1)

    def __getitem__(self, idx):
        return self._vector[idx - 1]


def test_matrix_keys():
    """FeatureMatrix from key iterator"""
    matrix = _pyliblinear.FeatureMatrix(
        [(3, KeyIterator((5, 8))), (4, KeyIterator((1, 3, 6)))]
    )

    assert matrix.width == 3
    assert matrix.height == 2
    assert list(matrix.labels()) == [3.0, 4.0]
    assert list(matrix.features()) == [
        {1: 5.0, 2: 8.0},
        {1: 1.0, 2: 3.0, 3: 6.0},
    ]


def test_matrix_keys_assign():
    """FeatureMatrix from key iterator and assigned labels"""
    matrix = _pyliblinear.FeatureMatrix(
        [KeyIterator((5, 8)), KeyIterator((1, 3, 6))],
        assign_labels=2.0,
    )

    assert matrix.width == 3
    assert matrix.height == 2
    assert list(matrix.labels()) == [2.0, 2.0]
    assert list(matrix.features()) == [
        {1: 5.0, 2: 8.0},
        {1: 1.0, 2: 3.0, 3: 6.0},
    ]


def test_matrix_from_iterables_dict():
    """FeatureMatrix.from_iterables from dicts"""
    matrix = _pyliblinear.FeatureMatrix.from_iterables(
        [2, 3],
        [{3: 4, 1: 7}, {2: 1}],
    )

    assert matrix.width == 3
    assert matrix.height == 2
    assert list(matrix.labels()) == [2.0, 3.0]
    assert list(matrix.features()) == [{1: 7.0, 3: 4.0}, {2: 1.0}]


def test_matrix_from_iterables_exc():
    """FeatureMatrix.from_iterables raises exception on different lengths"""
    with raises(ValueError):
        _pyliblinear.FeatureMatrix.from_iterables(
            [2, 3],
            [{3: 4, 1: 7}, {2: 1}, {2, 1}],
        )
