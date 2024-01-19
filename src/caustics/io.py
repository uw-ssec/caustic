from pathlib import Path

DEFAULT_ENCODING = "utf-8"


def to_file(
    path: str | Path, data: str | bytes, encoding: str = DEFAULT_ENCODING
) -> str:
    """
    Save data string or bytes to specified file path

    Parameters
    ----------
    path : str or Path
        The path to save the data to
    data : str | bytes
        The data string or bytes to save to file
    encoding : str, optional
        The string encoding to use, by default "utf-8"

    Returns
    -------
    str
        The path string where the data is saved
    """
    # TODO: Update to allow for remote paths saving

    # Convert string data to bytes
    if isinstance(data, str):
        data = data.encode(encoding)

    # Convert string path to Path object
    if isinstance(path, str):
        path = Path(path)

    path.write_bytes(data)
    return str(path.absolute())
