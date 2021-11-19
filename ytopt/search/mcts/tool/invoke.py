#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import io
import threading
import platform
import pathlib
from tool.support import *
import datetime


def empty_none(arg):
    if arg is None:
        return []
    return arg


class Invoke:
    # If called program fails, throw an exception
    EXCEPTION = NamedSentinel('EXCEPTION')

    # If called program fails, exit with same error code
    ABORT =  NamedSentinel('ABORT')

    # Like ABORT, but also print that the program has failed
    ABORT_EXITCODE =  NamedSentinel('ABORT_EXITCODE')

    # If called program fails, ignore or return error code
    IGNORE = NamedSentinel('IGNORE')


    @staticmethod
    def hlist(arg):
        if arg is None:
            return []
        if isinstance(arg, str) or isinstance(arg,pathlib.PurePath): # A filename
            return [arg]
        return list(arg)


    @staticmethod
    def assemble_env(env=None,setenv=None,appendenv=None):
        if env is None:
            env = dict(os.environ)
        if setenv is not None:
            for key,val in setenv.items():
                env[str(key)] = str(val)
        if appendenv is not None:
            for key,val in appendenv.items():
                if not val:
                    continue
                oldval = env.get(key)
                if oldval:
                    env[key] = oldval + ':' + val
                else:
                    env[key] = val
        return env


    def __init__(self,cmd,*args,cwd=None,setenv=None,appendenv=None,stdout=None,stderr=None,std_joined=None,std_prefixed=None):
        self.cmdline = [str(cmd)] + [str(a) for a in args]
        self.cwd = cwd
        self.setenv = setenv
        self.appendenv = appendenv

        self.stdout =Invoke. hlist(stdout)
        self.stderr = Invoke.hlist(stderr)
        self.std_joined=Invoke.hlist(std_joined)
        self.std_prefixed=Invoke.hlist(std_prefixed)


    def cmd(self):
        # FIXME: Windows
        shortcmd = self.cmdline[0]
        args = self.cmdline[1:]
        envs = []
        if self.setenv is not None:
            for envkey,envval in self.setenv.items():
                envs += [envkey + '=' + shquote(envval)]
        if self.appendenv is not None:
            for envkey,envval in self.appendenv.items():
                if not envval:
                    continue
                envs += [envkey + '=${' + envkey + '}:' + shquote(envval)]
        if envs :
          env =  ' '.join(envs)+' '
        else:
          env = ''
        if self.cwd is None:
             result  =  env + shortcmd + (' ' if args else '') + shjoin(args)
        else:
             result =  '(cd ' + shquote(self.cwd) + ' && ' + env + shortcmd + (' ' if args else '') + shjoin(args) + ')'
        return result


    def execute(self,onerror=EXCEPTION,timeout=None,
                print_stdout=False,print_stderr=False,print_prefixed=False,print_command=False,print_exitcode=False,
                return_exitcode=False,return_stdout=False,return_stderr=False,return_joined=False,return_prefixed=False,
                stdout=None,stderr=None,std_joined=None,std_prefixed=None):
        cmdline = [str(s) for s in self.cmdline]
        cwd = None if (self.cwd is None) else str(self.cwd)
        env = Invoke.assemble_env(setenv=self.setenv,appendenv=self.appendenv) if self.setenv or self.appendenv else None
        stdin=None

        if print_command or return_prefixed or std_prefixed:
            command = self.cmd()
        if print_command:
            print('$',command,flush=True)

        if isinstance(timeout,datetime.timedelta):
            timeout = timeout.total_seconds()

        stdouthandles = []
        stderrhandles = []
        prefixedhandles = []
        handlestoclose = []

        if print_stdout:
            stdouthandles.append(sys.stdout)
        if print_stderr:
            stderrhandles.append(sys.stderr)

        result_stdout = None
        if return_stdout:
            result_stdout = io.StringIO()
            stdouthandles.append(result_stdout)
        result_stderr = None
        if return_stderr:
            result_stderr = io.StringIO()
            stderrhandles.append(result_stderr)
        result_joined = None
        if return_joined:
            result_joined = io.StringIO()
            stdouthandles.append(result_joined)
            stderrhandles.append(result_joined)
        result_prefixed=None
        if return_prefixed:
            result_prefixed = io.StringIO()
            prefixedhandles.append(result_prefixed)

        def open_or_handle(arg):
                nonlocal handlestoclose
                if isinstance(arg,str) or isinstance(arg,pathlib.PurePath):
                    h = open(arg, 'w')
                    handlestoclose.append(h)
                    return h
                return arg


        for file in self.stdout + Invoke.hlist(stdout):
            stdouthandles.append(open_or_handle(file))
        for file in self. stderr +Invoke.hlist(stderr):
            stderrhandles.append(open_or_handle(file))
        for file in self. std_joined +Invoke.hlist(std_joined):
            h = open_or_handle(file)
            stdouthandles.append(h)
            stderrhandles.append(h)
        for file in self. std_prefixed +Invoke.hlist(std_prefixed):
            prefixedhandles.append(open_or_handle(file))



        for h in prefixedhandles:
            print(command,file=h)

        stdout_mode = subprocess.PIPE if stdouthandles or prefixedhandles else None
        stderr_mode = subprocess.PIPE if stderrhandles or prefixedhandles else None

        errmsg = None
        start = datetime.datetime.now()

        try:
          # bufize=1 ensures line buffering
          p = subprocess.Popen(cmdline,cwd=cwd,env=env,stdout=stdout_mode,stderr=stderr_mode,universal_newlines=True,bufsize=1)
        except Exception as err:
           # Exception can happen if e.g. the excutable does not exist
           if onerror is Invoke.EXCEPTION:
             raise

           # Process it further down
           p = None
           exitcode = 127 # Not sure whether this is sh/Linux-specific
           errmsg = "Invocation error: {err}".format(err=err)

        def catch_std(std, outhandles, prefix, prefixedhandles):
            while True:
                try:
                    line = std.readline()
                except ValueError as e:
                    # TODO: Handle properly
                    print("Input prematurely closed")
                    break

                if line is None or len(line) == 0:
                    break
                for h in outhandles:
                    print(line,end='',file=h)
                for h in prefixedhandles:
                    print(prefix,line,sep='',end='',file=h)

        if p:
          if stdout_mode == subprocess.PIPE:
            tout = threading.Thread(target=catch_std,
                                    kwargs={'std': p.stdout, 'outhandles': stdouthandles, 'prefix': "[stdout] ", 'prefixedhandles': prefixedhandles})
            tout.daemon = True
            tout.start()

          if stderr_mode == subprocess.PIPE:
            terr = threading.Thread(target=catch_std,
                                    kwargs={'std': p.stderr, 'outhandles': stderrhandles, 'prefix': "[stderr] ", 'prefixedhandles': prefixedhandles})
            terr.daemon = True
            terr.start()

          if stdin is not None:
             p.communicate(input=stdin)

          killed = False
          try:
            exitcode = p.wait(timeout=timeout)
          except subprocess.TimeoutExpired as e:
            # Kill the process after timeout
            p.kill()
            killed = e
            exitcode = None
            errmsg = "Timeout exceeded"

        stop = datetime.datetime.now()
        walltime = stop - start

        if p:
          if stdout_mode == subprocess.PIPE:
            tout.join()
            try:
                p.stdout.close()
            except:
                # Stale file handle possible
                # FIXME: why?
                pass

          if stderr_mode == subprocess.PIPE:
            terr.join()
            try:
                p.stderr.close()
            except:
                # Stale file handle possible
                # FIXME: Why?
                pass

        if errmsg:
          exitmsg = "Invocation failed in {walltime}: {errmsg}".format(exitcode=exitcode,walltime=walltime,errmsg=errmsg)
        else:
          exitmsg = "Exit with code {exitcode} in {walltime}".format(exitcode=exitcode,walltime=walltime)
        for h in prefixedhandles:
            print(exitmsg.format(rtncode=exitcode),file=h)
        if print_exitcode:
            print(exitmsg)

        if exitcode and onerror is Invoke.ABORT_EXITCODE:
                if errmsg:
                  exitmsg = "Command failed with code {rtncode}\n{errmsg}\n{command}".format(rtncode=exitcode,walltime=walltime,command=self.cmd(),errmsg=errmsg)
                else:
                  exitmsg = "Command failed with code {rtncode}\n{command}".format(rtncode=exitcode,walltime=walltime,command=self.cmd())
                for h in stderrhandles:
                    h.write(exitmsg)

        # TODO: Use context manager
        # TODO: if executing asynchronously, still need to close handles
        for h in handlestoclose:
            h.close()

        if exitcode or killed:
            if onerror is Invoke.ABORT or onerror is Invoke.ABORT_EXITCODE:
               # Treat as if the error was raised by this python program
               if killed:
                   exit(1)
               exit(exitcode)
            elif onerror is Invoke.EXCEPTION:
               # Allow application to handle this error
               # TODO: Mimic subprocess.check_output()
               if killed:
                   raise killed
               raise subprocess.CalledProcessError(returncode=exitcode, cmd=shjoin(cmdline))


        if isinstance(result_stdout, io.StringIO):
            result_stdout = result_stdout.getvalue()
        if isinstance(result_stderr, io.StringIO):
            result_stderr = result_stderr.getvalue()
        if isinstance(result_joined, io.StringIO):
            result_joined = result_joined.getvalue()
        if isinstance(result_prefixed, io.StringIO):
            result_prefixed = result_prefixed.getvalue()

        class InvokeResult:
            def __init__(self,exitcode,stdout,stderr,joined,prefixed,walltime):
                self.exitcode = exitcode
                self.stdout = stdout
                self.stderr = stderr
                self.joined = joined
                self.prefixed = prefixed
                self.walltime = walltime

            @property
            def success(self):
              return self.exitcode==0

            @property
            def output(self):
                 return self.prefixed or self.joined or self.stdout or self.stderr

        return InvokeResult(exitcode=exitcode,stdout=result_stdout,stderr=result_stderr,joined=result_joined,prefixed=result_prefixed,walltime=walltime)


    # Execute as if this is the command itself
    def run(self,onerror=None,print_stdout=True,print_stderr=True,**kwargs):
        return self.execute(onerror=first_defined(onerror, Invoke.ABORT),return_exitcode=True,print_stdout=print_stdout,print_stderr=print_stderr, **kwargs).exitcode

    # Diagnostic mode, execute with additional info
    def diag(self,onerror=None,print_stdout=False,print_stderr=False,print_prefixed=True,print_command=True,print_exitcode=True,**kwargs):
      return self.execute(onerror=first_defined(onerror, Invoke.IGNORE),return_exitcode=True,print_stdout=print_stdout,print_stderr=print_stderr,print_prefixed=print_prefixed,print_command=print_command,print_exitcode=True,**kwargs)

    # Execute to get the command's result
    def query(self,onerror=None,**kwargs):
        return self.execute(onerror=first_defined(onerror, Invoke.EXCEPTION),return_stdout=True, **kwargs).stdout

    def call(self,onerror=None,**kwargs):
        return self.execute(onerror=first_defined(onerror, Invoke.EXCEPTION),**kwargs)



def execute(cmd,*args,cwd=None,setenv=None,appendenv=None, **kwargs):
  return Invoke(cmd, *args, cwd=cwd,setenv=setenv,appendenv=appendenv).execute(**kwargs)

def run(cmd,*args,cwd=None,setenv=None,appendenv=None,**kwargs):
  return Invoke(cmd, *args,  cwd=cwd,setenv=setenv,appendenv=appendenv).run(**kwargs)

def diag(cmd,*args,cwd=None,setenv=None,appendenv=None, **kwargs):
  return Invoke(cmd, *args,  cwd=cwd,setenv=setenv,appendenv=appendenv).diag(**kwargs)

def query(cmd,*args,cwd=None,setenv=None,appendenv=None,  **kwargs):
  return Invoke(cmd, *args,  cwd=cwd,setenv=setenv,appendenv=appendenv).query(**kwargs)

def call(cmd,*args,cwd=None,setenv=None,appendenv=None,  **kwargs):
  return Invoke(cmd, *args,  cwd=cwd,setenv=setenv,appendenv=appendenv).call(**kwargs)





