#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from tool.support import *


class SupportTesting(unittest.TestCase):
        def test_shsplit(self):
           self.assertEqual(shsplit('a b'), ['a', 'b'])
           self.assertEqual(shsplit("A B C"), ['A', 'B', 'C'])

        def test_shquote(self):
          self.assertEqual(shquote('a "b',windows=True), '"a ""b"')
          self.assertEqual(shquote('a "b',windows=False), '\'a "b\'')
          self.assertEqual(shquote("ABC"), "ABC")
          self.assertEqual(shquote("A B C",windows=False), "'A B C'")
          self.assertEqual(shquote('A"B C',windows=True), '"A""B C"')
          self.assertEqual(shquote('A"B C',windows=False), '\'A"B C\'')


        def test_shjoin(self):
          self.assertEqual(shjoin(['a b', 'c'],windows=True), '"a b" c')
          self.assertEqual(shjoin(['a b', 'c'],windows=False), '\'a b\' c')
          self.assertEqual(shjoin(['A', ' B ', 'C'], windows=False), "A ' B ' C")
          self.assertEqual(shjoin(['A', ' B ', 'C'], windows=False, sep='+'), "A+' B '+C")
          self.assertEqual(shjoin(['A', ' B ', 'C'], windows=True), 'A " B " C')

        def test_versioncmp(self):
          self.assertTrue(version_cmp('1.5', '1.4'))
          self.assertTrue(version_cmp('1.5', '1.5'))
          self.assertFalse(version_cmp('1.4', '1.5'))

        def test_first_defined(self):
          self.assertEqual(first_defined(), None)
          self.assertEqual(first_defined(False), False)
          self.assertEqual(first_defined(True), True)
          self.assertEqual(first_defined(None, ''), '')
          self.assertEqual(first_defined(None, '', [], 0, 0.0,(), {}, dict(),set(), fallback='c'), '')
          self.assertEqual(first_defined(None, 'a', fallback='c'), 'a')



        def test_predefined(self):
          self.assertEqual(predefined('vc', {'vc': 'replacement'}), ('replacement',1) )
          self.assertEqual(predefined('VC', {'vc': 'replacement'}), ('replacement',1) )
          self.assertEqual(predefined('vc', {'vc'}), ('vc',1) )
          self.assertEqual(predefined('VC', {'vc'}), ('vc',1) )
          self.assertEqual(predefined('VC', {'vc': 'VC'}), ('VC',1) )
          self.assertEqual(predefined('VC', {'vc'},{ 'vf': 'replacement'}), ('vc',1) )
          self.assertEqual(predefined('VC', {'vc'},{ 'vf': 'replacement'}), ('vc',1) )
          self.assertEqual(predefined('VC', {'vc': 'cf'}, {'cf': 'replacement'}), ('replacement',2) )
          self.assertEqual(predefined('VC', {'vc': 'cf', 'cf': 'replacement'}), ('cf',1) )
          self.assertEqual(predefined('VC', {'vc': 'cf'}, {'cf'}), ('cf',2) )

          self.assertEqual(predefined('vc'), ('vc',True) )
          self.assertEqual(predefined('a', {'b'}), ('a',True) )
          self.assertEqual(predefined('c', default='v'), ('c',True) )
          self.assertEqual(predefined('c', default=None), ('c',True) )
          self.assertEqual(predefined('c', default='v'), ('c',True) )
          self.assertEqual(predefined('c', {'v': 'x'}, default='v'), ('c',True) )
          self.assertEqual(predefined(None, default='v'), ('v',False) )
          self.assertEqual(predefined(None, default=None), (None,False) )

        def test_predefined_fallback(self):
          self.assertEqual(predefined_fallback('vc', {'vc': 'replacement'}), ('replacement',1) )
          self.assertEqual(predefined_fallback('a'), (None,False) )
          self.assertEqual(predefined_fallback('a',fallback='b'), ('b',False) )
          self.assertEqual(predefined_fallback('x', {'y': 'replacement'},fallback='y'), ('replacement',False) )

        def test_predefined_strict(self):
          self.assertEqual(predefined_strict('a', {'a': 'b'}), 'b' )
          self.assertEqual(predefined_strict(None,default=None), None )
          with self.assertRaises(Exception):
            predefined_strict('b', {'a': 'b'})
          with self.assertRaises(Exception):
            predefined_strict(None)


if __name__ == '__main__':
    import tracemalloc
    tracemalloc.start()
    import sys
    unittest.main(argv=sys.argv[:1]) # Ignore arguments
    print("All tests passed")
