import os
import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import multiprocess
import multiprocessing

from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece):
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
            raise RuntimeError()
        exec(code_piece, self._global_vars)

    def eval_code(self, expr):
        return eval(expr, self._global_vars)

    def inject(self, var_dict):
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars['answer']


class PythonExecutor:
    def __init__(self, runtime=None, get_answer_symbol=None, get_answer_expr=None, get_answer_from_stdout=False,
                 timeout_length=5, ):
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.timeout_length = timeout_length

    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]

    @staticmethod
    def execute(code, get_answer_from_stdout=None, runtime=None, answer_symbol=None, answer_expr=None,
                timeout_length=10):
        try:
            if get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
            else:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code[:-1]))
                result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            report = "Done"
            str(result)
            pickle.dumps(result)
        except:
            result = ''
            report = traceback.format_exc().split('\n')[-2]
        return result, report

    def apply(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        with ProcessPool(max_workers=min(len(all_code_snippets), os.cpu_count())) as pool:
            executor = partial(
                self.execute,
                get_answer_from_stdout=self.get_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expr=self.answer_expr,
                timeout_length=self.timeout_length,  # this timeout not work
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            if len(all_code_snippets) > 100:
                progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
            else:
                progress_bar = None

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    exit()
                if progress_bar is not None:
                    progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post process
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results
