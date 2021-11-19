import unittest
import sys
from tool.invoke import *


class InvokeTesting(unittest.TestCase):
  def test_funcs(self):
           py3 = sys.executable
           self.assertEqual(execute(py3, '-c', 'print("X Y Z")').exitcode, 0)
           self.assertEqual(run(py3, '-c', 'print("X Y Z")'), 0)
           self.assertEqual(query(py3, '-c', 'print("X Y Z")'), 'X Y Z\n')
           with self.assertRaises(subprocess.CalledProcessError):
               call(py3, '-c', 'exit(42)')


  def test_invoke(self):
            import tempfile
            py3 = sys.executable

            # Test successful execution
            invokeXYZ = Invoke(py3, '-c', 'print("X Y Z")')
            self.assertEqual(invokeXYZ.run(), 0)

            # Test reusability of invoke object
            self.assertEqual(invokeXYZ.run(), 0)

            # Test changing directory
            with tempfile.TemporaryDirectory(suffix='cwd') as dir:
              invokeCwd = Invoke(py3, '-c', 'import os; print(os.getcwd(),end="")', cwd=dir)
              self.assertEqual(invokeCwd.query(), dir)

            # Test setenv
            invokeSetenv = Invoke(py3, '-c', 'import os; print(os.getenv("TESTME"))', setenv={'TESTME': 'A B C'})
            self.assertEqual(invokeSetenv.query(), 'A B C\n')

            # Test appendenv
            copath = os.getenv('PATH')
            invokeAppendenv = Invoke(py3, '-c', 'import os; print(os.getenv("PATH"))', appendenv={'PATH': 'A B C'})
            self.assertEqual(invokeAppendenv.query(), copath + ':A B C\n')

            # Test captured stdout
            self.assertEqual(invokeXYZ.query(), 'X Y Z\n')

            # Test captured stderr
            invokeXYZerr = Invoke(py3, '-c', 'import sys; print("X Y Z",file=sys.stderr)')
            self.assertEqual(invokeXYZerr.execute(return_stderr=True).output, 'X Y Z\n')

            # Test capture of stdout+stderr (order of output is undefined)
            invokeXYZouterr = Invoke(py3, '-c', 'import sys; print("A B C"); print("X Y Z",file=sys.stderr)')
            joined = invokeXYZouterr.execute(return_joined=True).output
            self.assertRegex(joined, 'A\ B\ C\n')
            self.assertRegex(joined, 'X\ Y\ Z\n')

            # Test capture of stdout+stderr with prefix
            joined = invokeXYZouterr.execute(return_prefixed=True).output
            self.assertRegex(joined, 'import\ sys') # Replicate command line at the beginning
            self.assertRegex(joined, '\[stdout\]\ A\ B\ C\n')
            self.assertRegex(joined, '\[stderr\]\ X\ Y\ Z')
            self.assertRegex(joined, 'Exit with code 0') # Emit exit code at the end

            # Test capture of everything at once
            with tempfile.TemporaryFile(suffix='out',mode='w+') as stream_out,tempfile.TemporaryFile(suffix='err',mode='w+') as stream_err,tempfile.TemporaryFile(suffix='joined',mode='w+') as stream_joined,tempfile.TemporaryFile(suffix='prefixed',mode='w+') as stream_prefixed:
              fd_out,file_out = tempfile.mkstemp(suffix='out')
              fd_err,file_err = tempfile.mkstemp(suffix='err')
              fd_joined,file_joined = tempfile.mkstemp(suffix='joined')
              fd_prefixed,file_prefixed = tempfile.mkstemp(suffix='prefixed')

              captureall = invokeXYZouterr.execute(print_stdout=True,print_stderr=True, print_prefixed=True,
                                             return_stdout=True, return_stderr=True, return_joined=True, return_prefixed=True,
                                             stdout=[stream_out,file_out],stderr=[stream_err,file_err],std_joined=[stream_joined,file_joined],std_prefixed=[stream_prefixed,file_prefixed])

              def readstream(stream):
                   stream.seek(0)
                   return stream.read()

              def readfile(fd, fname):
                with open(fname, 'r') as content_file:
                  result = content_file.read()
                os.close(fd)
                os.remove(fname)
                return result

              self.assertEqual(captureall.stdout, 'A B C\n')
              self.assertEqual(readstream(stream_out), 'A B C\n')
              self.assertEqual(readfile(fd_out,file_out), 'A B C\n')

              self.assertEqual(captureall.stderr, 'X Y Z\n')
              self.assertEqual(readstream(stream_err), 'X Y Z\n')
              self.assertEqual(readfile(fd_err,file_err), 'X Y Z\n')

              self.assertRegex(captureall.joined, 'A\ B\ C\n')
              self.assertRegex(captureall.joined, 'X\ Y\ Z\n')
              self.assertEqual(readstream(stream_joined), captureall.joined)
              self.assertEqual(readfile(fd_joined,file_joined), captureall.joined)

              self.assertRegex(captureall.prefixed, '\[stdout\]\ A\ B\ C\n')
              self.assertRegex(captureall.prefixed, '\[stderr\]\ X\ Y\ Z\n')
              self.assertEqual(readstream(stream_prefixed), captureall.prefixed)
              self.assertEqual(readfile(fd_prefixed,file_prefixed), captureall.prefixed)



            # Test failing program
            invokeError = Invoke(py3, '-c', """exit(42)""")
            self.assertEqual(invokeError.run(onerror=Invoke.IGNORE), 42)

            # Failing program throwing an exception
            with self.assertRaises(subprocess.CalledProcessError):
               invokeError.execute()

            # Test ABORT
            with self.assertRaises(SystemExit) as cm:
              invokeError.run()
            self.assertEqual(cm.exception.code, 42)

            # Test ABORT_EXITCODE (should print message to stderr)
            with self.assertRaises(SystemExit) as cm:
              invokeError.run(onerror=Invoke.ABORT_EXITCODE)
            self.assertEqual(cm.exception.code, 42)


  def test_win_escape(self):
            py3 = sys.executable

            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', 'a^{commit}').query(), 'a^{commit}\n')
            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', '{}').query(), '{}\n')

            # Test caret (special character on windows cmd)
            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', 'a').query(), 'a\n')
            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', '^a').query(), '^a\n')
            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', 'a^').query(), 'a^\n')
            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', '^^a').query(), '^^a\n')
            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', 'a^^').query(), 'a^^\n')
            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', '"^a"').query(), '"^a"\n')
            self.assertEqual(Invoke(py3, '-c', 'import sys; print(sys.argv[1])', '"a^"').query(), '"a^"\n')





if __name__ == '__main__':
    import tracemalloc
    tracemalloc.start()

    #InvokeTesting().test_win_escape()

    unittest.main(argv=sys.argv[:1]) # Ignore arguments
    print("All tests passed")
