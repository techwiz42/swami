import inspect
import cerebrum
from typing import Any, Dict, List, Set, Optional
import typing
import sys
from pprint import pprint

def get_full_type_str(type_hint: Any) -> str:
    """Get a readable string representation of a type hint."""
    if type_hint == inspect.Parameter.empty:
        return 'Any'
    return str(type_hint)

def analyze_type_hints(obj: Any) -> Dict[str, str]:
    """Analyze type hints for a class or function."""
    if not hasattr(obj, '__annotations__'):
        return {}
    
    return {
        name: get_full_type_str(hint) 
        for name, hint in obj.__annotations__.items()
    }

def analyze_parameter(param: inspect.Parameter) -> Dict[str, Any]:
    """Analyze a single parameter of a function."""
    return {
        'name': param.name,
        'kind': str(param.kind),
        'default': str(param.default) if param.default != inspect.Parameter.empty else None,
        'annotation': get_full_type_str(param.annotation),
    }

def analyze_function_detailed(func) -> Dict[str, Any]:
    """Analyze a function in detail, including its signature, docstring, and type hints."""
    sig = inspect.signature(func)
    
    return {
        'name': func.__name__,
        'qualname': func.__qualname__,
        'module': func.__module__,
        'doc': inspect.getdoc(func),
        'annotations': analyze_type_hints(func),
        'parameters': [
            analyze_parameter(param) 
            for param in sig.parameters.values()
        ],
        'return_annotation': get_full_type_str(sig.return_annotation),
        'is_coroutine': inspect.iscoroutinefunction(func),
        'source_file': inspect.getfile(func) if hasattr(func, '__code__') else None,
    }

def analyze_class_detailed(cls: type) -> Dict[str, Any]:
    """Analyze a class in detail, including its methods, attributes, and inheritance."""
    # Get all members including inherited ones
    all_members = inspect.getmembers(cls)
    
    # Get direct class attributes (not inherited)
    class_attrs = {
        name: value for name, value in cls.__dict__.items()
        if not name.startswith('_')
    }
    
    # Analyze methods
    methods = {}
    for name, member in all_members:
        if name.startswith('_'):
            continue
        if inspect.isfunction(member) or inspect.ismethod(member):
            methods[name] = analyze_function_detailed(member)
    
    # Get bases (inheritance)
    bases = [
        base.__name__ for base in cls.__bases__
        if base != object
    ]
    
    return {
        'name': cls.__name__,
        'qualname': cls.__qualname__,
        'module': cls.__module__,
        'doc': inspect.getdoc(cls),
        'bases': bases,
        'methods': methods,
        'attributes': {
            name: str(type(value).__name__)
            for name, value in class_attrs.items()
            if not callable(value)
        },
        'annotations': analyze_type_hints(cls),
        'source_file': inspect.getfile(cls),
    }

def analyze_module_detailed(module: Any, visited: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Analyze a module and all its submodules in detail."""
    if visited is None:
        visited = set()
    
    # Avoid circular imports
    if module.__name__ in visited:
        return {}
    visited.add(module.__name__)
    
    module_info = {
        'name': module.__name__,
        'doc': inspect.getdoc(module),
        'file': getattr(module, '__file__', None),
        'classes': {},
        'functions': {},
        'submodules': {},
        'constants': {},
    }
    
    for name, obj in inspect.getmembers(module):
        # Skip private/special members
        if name.startswith('_'):
            continue
            
        # Analyze based on type
        if inspect.isclass(obj):
            module_info['classes'][name] = analyze_class_detailed(obj)
        elif inspect.isfunction(obj):
            module_info['functions'][name] = analyze_function_detailed(obj)
        elif inspect.ismodule(obj) and obj.__name__.startswith(module.__name__):
            module_info['submodules'][name] = analyze_module_detailed(obj, visited)
        elif not callable(obj) and not inspect.ismodule(obj):
            module_info['constants'][name] = str(type(obj).__name__)
    
    return module_info

def print_detailed_api(api_info: Dict[str, Any], indent: int = 0) -> None:
    """Print the detailed API structure in a readable format."""
    indent_str = '  ' * indent
    
    print(f"{indent_str}Module: {api_info['name']}")
    if api_info['doc']:
        print(f"{indent_str}Documentation:\n{indent_str}{api_info['doc']}\n")
    
    if api_info['constants']:
        print(f"{indent_str}Constants:")
        for name, type_name in api_info['constants'].items():
            print(f"{indent_str}  {name}: {type_name}")
    
    if api_info['classes']:
        print(f"\n{indent_str}Classes:")
        for class_name, class_info in api_info['classes'].items():
            print(f"\n{indent_str}  {class_name}:")
            if class_info['bases']:
                print(f"{indent_str}    Inherits from: {', '.join(class_info['bases'])}")
            if class_info['doc']:
                print(f"{indent_str}    Doc: {class_info['doc']}")
            
            if class_info['attributes']:
                print(f"{indent_str}    Attributes:")
                for attr_name, attr_type in class_info['attributes'].items():
                    print(f"{indent_str}      {attr_name}: {attr_type}")
            
            if class_info['methods']:
                print(f"{indent_str}    Methods:")
                for method_name, method_info in class_info['methods'].items():
                    params = [f"{p['name']}: {p['annotation']}" for p in method_info['parameters']]
                    print(f"{indent_str}      {method_name}({', '.join(params)}) -> {method_info['return_annotation']}")
                    if method_info['doc']:
                        print(f"{indent_str}        {method_info['doc']}")
    
    if api_info['functions']:
        print(f"\n{indent_str}Functions:")
        for func_name, func_info in api_info['functions'].items():
            params = [f"{p['name']}: {p['annotation']}" for p in func_info['parameters']]
            print(f"{indent_str}  {func_name}({', '.join(params)}) -> {func_info['return_annotation']}")
            if func_info['doc']:
                print(f"{indent_str}    {func_info['doc']}")
    
    if api_info['submodules']:
        print(f"\n{indent_str}Submodules:")
        for submodule_info in api_info['submodules'].values():
            print_detailed_api(submodule_info, indent + 1)

if __name__ == '__main__':
    print("Analyzing Cerebrum API structure in detail...")
    api_info = analyze_module_detailed(cerebrum)
    
    print("\nCerebrum Detailed API Structure:")
    print("=" * 80)
    print_detailed_api(api_info)
    
    # Also save the raw data structure for further analysis
    with open('cerebrum_api.json', 'w') as f:
        import json
        json.dump(api_info, f, indent=2)
    print("\nDetailed API information has been saved to 'cerebrum_api.json'")
