from collections import abc
from typing import Callable, Sequence, Tuple, Union

from megengine.utils.tuple_function import _pair, _quadruple, _single, _triple

from ..types import Number

to_1tuple: Callable[[Union[Sequence[Number], Number]], Tuple[int]] = _single
to_2tuple: Callable[[Union[Sequence[Number], Number]], Tuple[int, int]] = _pair
to_3tuple: Callable[[Union[Sequence[Number], Number]], Tuple[int, int, int]] = _triple
to_4tuple: Callable[
    [Union[Sequence[Number], Number]], Tuple[int, int, int, int]
] = _quadruple


def is_seq_of(seq, expected_type, seq_type=None):
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    return is_seq_of(seq, expected_type, seq_type=tuple)


def add_doc(doc: str):
    """Use as a decorator"""

    def _warpper_func(target):
        target.__doc__ = doc
        return target

    return _warpper_func


def borrow_doc(origin):
    """Use as a decorator"""

    def _warpper_func(target):
        target.__doc__ = origin.__doc__
        return target

    return _warpper_func


def hack_module(self):
    self._modules = []
    self.name = None
    self._name = None
    self._short_name = None
