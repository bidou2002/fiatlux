# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:37:51 2018

@author: pjanin
"""


class Test1(object):

    def __init__(self, a):
        self._a = a

    def _get_a(self):
        return self._a

    def _set_a(self, a):
        self._a = a

    a = property(_get_a, _set_a)


class Test2(object):

    def __init__(self, a):
        self._a = a

    def __mul__(self, vararg):
        self._a = self._a * vararg._a
        return self

    def _get_a(self):
        return self._a

    def _set_a(self, a):
        self._a = a

    a = property(_get_a, _set_a)