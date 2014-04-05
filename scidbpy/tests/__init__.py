import os

from scidbpy import interface


def get_interface():
    """
    Connect to a SciDB instance by looking up the url
    in the SCIDB_URL environment variable, or defaulting
    to localhost
    """
    url = os.environ.get('SCIDB_URL', 'http://127.0.0.1:8080')
    return interface.SciDBShimInterface(url)
