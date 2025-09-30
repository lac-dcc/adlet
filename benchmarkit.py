import os
import csv
import shelve
import random
import logging
import subprocess
from abc import abstractmethod
from typing import Iterator, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class ShelveIterator:
    def __init__(self, db: shelve.Shelf, order: str = "sequential"):
        self._db = db
        self.order = order
        self._keys = list(self._db["configs"].keys())
        if self.order == "random":
            random.shuffle(self._keys)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, ...]:
        while self._index < len(self._keys):
            key = self._keys[self._index]
            self._index += 1
            if key in self._db["configs"]:
                return self._db["configs"][key]['params']

        raise StopIteration

class Benchmarkit:
    def __init__(self, command: str, recover_failure: bool = True, output_csv: str = 'output.csv', order: str ='sequential', reps: int = 10) -> None:
        self.recover = recover_failure
        self.output_csv = output_csv
        self.command = command
        self.order = order
        self.reps = reps
        self._db_name = '.benchmarkit.db'
        self._total: int = 0
        self._count: int = 0
        self._left: int = 0
        self._shelve_flag = 'n'
        self._db = None
        self.__load()

    def __load(self):
        if not self._can_recover():
            self._shelve_flag = 'n'
            self._db = shelve.open(self._db_name, flag=self._shelve_flag, writeback=True)
            self._build_dict()
        else:
            self._shelve_flag = 'c'
            self._db = shelve.open(self._db_name, flag=self._shelve_flag, writeback=True)
            try:
                self._total = self._db["total"]
                self._left = self._db["left"]
                self._count = self._db["count"]
            except KeyError:
                self._count = 0
                self._left = len(self._db["configs"]) * self.reps
                self._total = self._left

    def __del__(self):
        if self._db:
            self._db.close()


    def sync(self):
        if self._db:
            self._db["left"] = self._left
            self._db["count"] = self._count
            self._db["total"] = self._total
            self._db.sync()

    @abstractmethod
    def csv_header(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_output(self, output: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def gen_configs(self) -> Iterator[Tuple[str, ...]]:
        raise NotImplementedError


    def _gen_key(self, config: Tuple[str, ...]) -> str:
        return "#".join(map(str, config))

    def _build_dict(self):
        self._db["configs"] = {}
        for config in self.gen_configs():
            self._db["configs"][self._gen_key(config)] = {
                "params": config,
                "left": self.reps,
            }
        self._left = len(self._db["configs"]) * self.reps
        self._total = self._left
        self.sync()


    def _decrement(self, config: Tuple[str, ...], amount: int = 1) -> None:
        key = self._gen_key(config)
        cur_left = self._db["configs"][key]['left']
        left = cur_left - amount
        if left > 0:
            self._db["configs"][key]['left'] = left
        else:
            del self._db["configs"][key]
        self._left -= amount
        self._count += 1
        self.sync()

    def _can_recover(self) -> bool:
            return  os.path.exists(self.output_csv) and \
            os.path.exists(self._db_name) and self.recover

    def _get_file_flag(self) -> str:
        if self._can_recover():
            return 'a'
        return 'w'

    def run(self):

        file_flag = self._get_file_flag()

        with open(self.output_csv, file_flag, newline='') as file:
            writer = csv.writer(file)
            if file_flag == 'w':
                writer.writerow(self.csv_header())

            iterator = ShelveIterator(self._db, order=self.order)
            for config in iterator:
                for _ in range(self.reps):
                    logger.info(f"[{self._count}/{self._total}] running config {config}")
                    try:
                        cmd = [self.command]
                        cmd.extend(config)
                        process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        process.wait()
                        result: List[str] = self.parse_output(process.stdout.read())
                        self._decrement(config)
                        writer.writerow(result)
                        file.flush()
                    except Exception as e:
                        logger.error(f"Error running configuration {config}: {str(e)}")
                        raise e

class TACOBenchmark(Benchmarkit):
    def gen_configs(self) -> Iterator[Tuple[str, ...]]:
        import itertools
        benchmark = ["format"]
        sizes = list(map(str, [64, 128, 256, 512, 1024, 2048, 4096]))
        ratios = list(map(str, [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]))
        return itertools.product(benchmark, sizes, ratios)

    def parse_output(self, output):
        return output.split(",")

    def csv_header(self) -> str:
        return "M,N,K,time"


if __name__ == "__main__":
    sl = TACOBenchmark(command="./benchmark", recover_failure=True, reps=10)
    sl.run()
