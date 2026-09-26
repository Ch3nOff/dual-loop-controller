# imp compatibility layer for Python 3.12+ (specifically for flasgger)
import importlib.util
import os
import sys

def find_module(name, path=None):
    # flasgger uses: site_package = imp.find_module(path[0])[1]
    # It only cares about the second element of the return tuple (pathname).
    try:
        spec = importlib.util.find_spec(name, path)
    except Exception as e:
        raise ImportError(str(e))
        
    if spec is None:
        raise ImportError(f"No module named {name}")
    
    if spec.submodule_search_locations:
        pathname = list(spec.submodule_search_locations)[0]
    else:
        pathname = spec.origin
        
    # Return a dummy tuple with pathname at index 1
    return (None, pathname, None)

# Add to sys.modules so it behaves like a real module
sys.modules['imp'] = sys.modules[__name__]
