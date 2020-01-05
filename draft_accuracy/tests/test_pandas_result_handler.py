import pandas as pd
import pathlib

from prefect_ds import pandas_result_handler as prh


class TestInit:

    def test_extension_extracted_properly(self):
        filename = "a/b/c/d.cSv"
        handler = prh.PandasResultHandler(filename)
        assert handler.extension == "csv"

    def test_works_with_string_and_path(self):
        filename_string = "a/b/c/d.csv"
        filename_path = pathlib.Path(filename_string)
        handler_string = prh.PandasResultHandler(filename_string)
        handler_path = prh.PandasResultHandler(filename_path)
        assert handler_string.path == handler_path.path


class TestReadWrite:

    def test_read_write_works(self, tmp_path):
        filename = tmp_path / "test.csv"
        handler = prh.PandasResultHandler(filename, write_kwargs={"index": False})

        data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4, 5, 6]
        })
        handler.write(data)
        read_data = handler.read(data)
        pd.testing.assert_frame_equal(data, read_data)