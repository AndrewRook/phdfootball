import pandas as pd
import pathlib
import typing

from prefect.engine.result_handlers.result_handler import ResultHandler


class PandasResultHandler(ResultHandler):
    READ_OPS_MAPPING = {
        "csv": pd.read_csv
    }
    WRITE_OPS_MAPPING = {
        "csv": "to_csv"
    }

    def __init__(
            self,
            path: typing.Union[str, pathlib.Path],
            read_kwargs: dict = None,
            write_kwargs: dict = None
    ):
        self.path = pathlib.Path(path)

        self.extension = self.path.suffix.replace(".", "").lower()
        assert self.extension in self.READ_OPS_MAPPING and self.extension in self.WRITE_OPS_MAPPING
        self.read_kwargs = read_kwargs if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs if write_kwargs is not None else {}
        super().__init__()

    def read(self, _: typing.Optional = None, input_mapping=None) -> pd.DataFrame:
        input_mapping = {} if input_mapping is None else input_mapping
        path_string = str(self.path).format(**input_mapping)
        self.logger.debug("Starting to read result from {}...".format(path_string))
        data = self.READ_OPS_MAPPING[self.extension](
            path_string,
            **self.read_kwargs
        )
        self.logger.debug("Finished reading result from {}...".format(path_string))
        return data

    def write(self, result: pd.DataFrame, input_mapping=None):
        input_mapping = {} if input_mapping is None else input_mapping
        path_string = str(self.path).format(**input_mapping)
        self.logger.debug("Starting to write result to {}...".format(path_string))
        write_function = getattr(result, self.WRITE_OPS_MAPPING[self.extension])
        write_function(path_string, **self.write_kwargs)
        self.logger.debug("Finished writing result to {}...".format(path_string))
