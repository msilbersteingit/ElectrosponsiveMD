import unittest
import sys

sys.path.append('../../')
from analib import fileIO

class computeTest(unittest.TestCase):
    def test_check_file_does_not_exist(self):
        """Asserts NameError for non-existent file."""
        with self.assertRaises(NameError):
            fileIO.findInSubdirectory('nonefile')

    def test_check_file_exists(self):
        """Asserts file path for the given file"""
        return_path='./input\\test_dir\\test_file.txt'
        self.assertEqual(
            fileIO.findInSubdirectory(
                'test_file.txt','./'),
            return_path)
    
    def test_retrieve_filetypes(self):
        """Asserts file type returns for the given file"""
        rootname='test_file'
        dump_wrapped=rootname+'.lammpstrj'
        dump_unwrapped=rootname+'.unwrapped.lammpstrj'
        dump_def1=rootname+'.def1.txt'
        dump_def2=rootname+'.def2.txt'
        dump_def3=rootname+'.def3.txt'
        log_file='log.' + rootname + '.txt'
        self.assertEqual(fileIO.retrieve_different_filetypes('test_file'),
        (dump_wrapped, dump_unwrapped, dump_def1, 
        dump_def2, dump_def3, log_file))

if __name__=='__main__':
    unittest.main()