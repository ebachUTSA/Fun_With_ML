import abc
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)


class Singleton(abc.ABCMeta, type):
    """Singleton metaclass for ensuring only one instance of a class."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """Abstract singleton class used for future singlton classes."""


class Config(metaclass=Singleton):
    """Configuration class to store the state of bools for different scripts access."""

    def __init__(self):
        """Initialize the Config class."""
        self.alchemy_driver = os.getenv("ALCHEMY_DRIVER")
        self.pyodbc_driver = os.getenv("PYODBC_DRIVER")
        self.uid = os.getenv("UID")
        self.pid = os.getenv("PID")
        self.server = os.getenv("SERVER")
        self.port = os.getenv("PORT")
        self.database = os.getenv("DATABASE")
        self.base_directory = os.getenv("BASE_DIRECTORY")

        self._set_dynamic_attributes("DB")
        self._set_dynamic_attributes("TABLE")

    def _set_dynamic_attributes(self, suffix: str):
        for key, value in os.environ.items():
            if key.endswith(suffix):
                attr_name = key.lower()
                setattr(self, attr_name, value)

    def __repr__(self):
        """Representation of the Config class."""
        attrs = ",\n".join(
            f"    {attr}={repr(getattr(self, attr))}"
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        )
        return f"Config(\n{attrs}\n)"
