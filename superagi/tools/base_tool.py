import inspect
import os
from abc import abstractmethod
from functools import wraps
from inspect import signature
from typing import List
from typing import Optional, Type, Callable, Any, Union, Dict, Tuple
from enum import Enum
import yaml
from pydantic import BaseModel, create_model, validate_arguments, Extra

from superagi.types.key_type import ToolConfigKeyType
import os
from sqlalchemy.orm import Session
import csv
from superagi.helper.s3_helper import S3Helper
from superagi.lib.logger import logger
from superagi.config.config import get_config
from superagi.types.storage_types import StorageType


class SchemaSettings:
    """Configuration for the pydantic model."""
    extra = Extra.forbid
    arbitrary_types_allowed = True


def extract_valid_parameters(
        inferred_type: Type[BaseModel],
        function: Callable,
) -> dict:
    """Get the arguments from a function's signature."""
    schema = inferred_type.schema()["properties"]
    valid_params = signature(function).parameters
    return {param: schema[param] for param in valid_params if param != "run_manager"}


def _construct_model_subset(
        model_name: str, original_model: BaseModel, required_fields: list
) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    fields = {
        field: (
            original_model.__fields__[field].type_,
            original_model.__fields__[field].default,
        )
        for field in required_fields
        if field in original_model.__fields__
    }
    return create_model(model_name, **fields)  # type: ignore


def create_function_schema(
        schema_name: str,
        function: Callable,
) -> Type[BaseModel]:
    """Create a pydantic schema from a function's signature."""
    validated = validate_arguments(function, config=SchemaSettings)  # type: ignore
    inferred_type = validated.model  # type: ignore
    if "run_manager" in inferred_type.__fields__:
        del inferred_type.__fields__["run_manager"]
    valid_parameters = extract_valid_parameters(inferred_type, function)
    return _construct_model_subset(
        f"{schema_name}Schema", inferred_type, list(valid_parameters)
    )


class BaseToolkitConfiguration:

    def get_tool_config(self, key: str, module_dir):
        config_path = self.__find_config_file(module_dir)
        config = self.__load_config(config_path)
        return config.get(key)

    def __find_config_file(self, module_dir):
        root_dir = module_dir
        while True:
            config_path = os.path.join(root_dir, "config.yaml")
            if os.path.isfile(config_path):
                # Found the config.yaml file in the current directory
                return config_path
            parent_dir = os.path.dirname(root_dir)
            if parent_dir == root_dir:
                # Reached the root directory without finding the config.yaml file
                raise FileNotFoundError("config.yaml file not found")
            root_dir = parent_dir

    def __load_config(self, config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config


class BaseTool(BaseModel):
    name: str = None
    description: str
    args_schema: Type[BaseModel] = None
    permission_required: bool = True
    toolkit_config: BaseToolkitConfiguration = BaseToolkitConfiguration()

    class Config:
        arbitrary_types_allowed = True

    @property
    def args(self):
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        else:
            name = self.name
            args_schema = create_function_schema(f"{name}Schema", self.execute)
            return args_schema.schema()["properties"]

    @abstractmethod
    def _execute(self, *args: Any, **kwargs: Any):
        pass

    @property
    def max_token_limit(self):
        return 600

    def _parse_input(
            self,
            tool_input: Union[str, Dict],
    ) -> Union[str, Dict[str, Any]]:
        """Convert tool input to pydantic model."""
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                key_ = next(iter(input_args.__fields__.keys()))
                input_args.validate({key_: tool_input})
            return tool_input
        else:
            if input_args is not None:
                result = input_args.parse_obj(tool_input)
                return {k: v for k, v in result.dict().items() if k in tool_input}
        return tool_input

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        # For backwards compatibility, if run_input is a string,
        # pass as a positional argument.
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            return (), tool_input

    def execute(
            self,
            tool_input: Union[str, Dict],
            **kwargs: Any
    ) -> Any:
        """Run the tool."""
        parsed_input = self._parse_input(tool_input)

        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = (
                self._execute(*tool_args, **tool_kwargs)
            )
        except (Exception, KeyboardInterrupt) as e:
            raise e
        return observation

    @classmethod
    def from_function(cls, func: Callable, args_schema: Type[BaseModel] = None):
        if args_schema:
            return cls(description=func.__doc__, args_schema=args_schema)
        else:
            return cls(description=func.__doc__)

    def get_tool_config(self, key):
        caller_frame = inspect.currentframe().f_back
        caller_module = inspect.getmodule(caller_frame)
        caller_file = inspect.getfile(caller_module)
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        return self.toolkit_config.get_tool_config(key=key, module_dir=caller_dir)


class FunctionalTool(BaseTool):
    name: str = None
    description: str
    func: Callable
    args_schema: Type[BaseModel] = None

    @property
    def args(self):
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        else:
            name = self.name
            args_schema = create_function_schema(f"{name}Schema", self.execute)
            return args_schema.schema()["properties"]

    def _execute(self, *args: Any, **kwargs: Any):
        return self.func(*args, kwargs)

    @classmethod
    def from_function(cls, func: Callable, args_schema: Type[BaseModel] = None):
        if args_schema:
            return cls(description=func.__doc__, args_schema=args_schema)
        else:
            return cls(description=func.__doc__)

    def registerTool(cls):
        cls.__registerTool__ = True
        return cls


def tool(*args: Union[str, Callable], return_direct: bool = False,
         args_schema: Optional[Type[BaseModel]] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        nonlocal args_schema

        tool_instance = FunctionalTool.from_function(func, args_schema)

        @wraps(func)
        def wrapper(*tool_args, **tool_kwargs):
            if return_direct:
                return tool_instance._exec(*tool_args, **tool_kwargs)
            else:
                return tool_instance

        return wrapper

    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    else:
        return decorator


class BaseToolkit(BaseModel):
    name: str
    description: str

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        # Add file related tools object here
        pass

    @abstractmethod
    def get_env_keys(self) -> List[str]:
        # Add file related config keys here
        pass


class ToolConfiguration:

    def __init__(self, key: str, key_type: str = None, is_required: bool = False, is_secret: bool = False):
        self.key = key
        if is_secret is None:
            self.is_secret = False
        elif isinstance(is_secret, bool):
            self.is_secret = is_secret
        else:
            raise ValueError("is_secret should be a boolean value")
        if is_required is None:
            self.is_required = False
        elif isinstance(is_required, bool):
            self.is_required = is_required
        else:
            raise ValueError("is_required should be a boolean value")

        if key_type is None:
            self.key_type = ToolConfigKeyType.STRING
        elif isinstance(key_type, ToolConfigKeyType):
            self.key_type = key_type
        else:
            raise ValueError("key_type should be string/file/integer")
    
def get_resource_path( file_name: str):
        """Get final path of the resource.

        Args:
            file_name (str): The name of the file.
        """
        root_output_dir = get_root_output_dir() + file_name
        return root_output_dir

   
def get_root_output_dir():
        """Get root dir of the resource.
        """
        root_dir = get_config('RESOURCES_OUTPUT_ROOT_DIR')

        if root_dir is not None:
            root_dir = root_dir if root_dir.startswith("/") else os.getcwd() + "/" + root_dir
            root_dir = root_dir if root_dir.endswith("/") else root_dir + "/"
        else:
            root_dir = os.getcwd() + "/"
        return root_dir

    
class FileManager:
    def __init__(self, session: Session, agent_id: int = None, agent_execution_id: int = None):
        self.session = session
        self.agent_id = agent_id
        self.agent_execution_id = agent_execution_id
        
    def write_binary_file(self, file_name: str, data):
        if self.agent_id is not None:
            final_path = get_resource_path(file_name)
        else:
            final_path = get_resource_path(file_name)
        try:
            with open(final_path, mode="wb") as img:
                img.write(data)
                img.close()
            with open(final_path, 'rb') as img:
                storage_type = StorageType.get_storage_type(get_config("STORAGE_TYPE", StorageType.FILE.value))
                if  storage_type == StorageType.S3.value:
                    S3Helper().upload_file(img, path=final_path)
            logger.info(f"Binary {file_name} saved successfully")
            return f"Binary {file_name} saved successfully"
        except Exception as err:
            return f"Error write_binary_file: {err}"

    def write_file(self, file_name: str, content):
        if self.agent_id is not None:
            final_path = get_resource_path(file_name)
            
        else:
            final_path = get_resource_path(file_name)

        try:
            with open(final_path, mode="w") as file:
                file.write(content)
                file.close()
                
            with open(final_path, 'rb') as img:
                storage_type = StorageType.get_storage_type(get_config("STORAGE_TYPE", StorageType.FILE.value))
                if  storage_type == StorageType.S3.value:
                    S3Helper().upload_file(img, path=final_path)
            logger.info(f"{file_name} - File written successfully")
            return f"{file_name} - File written successfully"
        except Exception as err:
            return f"Error write_file: {err}"
        
    def write_csv_file(self, file_name: str, csv_data):
        if self.agent_id is not None:
            final_path = get_resource_path(file_name)
        else:
            final_path = get_resource_path(file_name)
        try:
            with open(final_path, mode="w", newline="") as file:
                writer = csv.writer(file, lineterminator="\n")
                writer.writerows(csv_data)
            with open(final_path, 'rb') as img:
                storage_type = StorageType.get_storage_type(get_config("STORAGE_TYPE", StorageType.FILE.value))
                if  storage_type == StorageType.S3.value:
                    S3Helper().upload_file(img, path=final_path)
            logger.info(f"{file_name} - File written successfully")
            return f"{file_name} - File written successfully"
        except Exception as err:
            return f"Error write_csv_file: {err}"
        
        
    def read_file(self, file_name: str):
        if self.agent_id is not None:
            final_path = get_resource_path(file_name)
        else:
            final_path = get_resource_path(file_name)

        try:
            with open(final_path, mode="r") as file:
                content = file.read()
            logger.info(f"{file_name} - File read successfully")
            return content
        except Exception as err:
            return f"Error while reading file {file_name}: {err}"
        
    def get_files(self):
        """
        Gets all file names generated by the CodingTool.
        Returns:
            A list of file names.
        """
        
        if self.agent_id is not None:
            final_path = "/assets/output/"
        else:
            final_path = "/assets/output/"
        try:
            # List all files in the directory
            files = os.listdir(final_path)
        except Exception as err:
            logger.error(f"Error while accessing files in {final_path}: {err}")
            files = []
        return files
     