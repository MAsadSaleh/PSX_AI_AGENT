"""
Patch for PSX data reader to handle 403 errors by adding proper headers to requests.
This file should be imported before using the PSX module.
"""

import sys
import os
import importlib
from pathlib import Path

def apply_patch():
    """Apply patches to PSX modules to prevent 403 errors"""
    try:
        # Get the location of psx module
        print("Attempting to patch PSX module...")
        
        # Try to locate the PSX module
        psx_module_found = False
        psx_module_path = None
        
        # Try different approaches to find the psx module
        try:
            import psx
            psx_module_path = os.path.dirname(psx.__file__)
            psx_module_found = True
            print(f"Found PSX module at: {psx_module_path}")
        except ImportError:
            print("PSX module not found in sys.path")
            
            # Check if it might be in the current directory structure
            current_dir = Path(__file__).parent
            potential_paths = [
                current_dir,
                current_dir.parent / "psx",
                Path("psx-data-reader-master") / "src" / "psx",
                Path("src") / "psx"
            ]
            
            for path in potential_paths:
                if (path / "__init__.py").exists():
                    psx_module_path = str(path)
                    psx_module_found = True
                    print(f"Found PSX module at: {psx_module_path}")
                    
                    # Add to sys.path if not already there
                    if str(path.parent) not in sys.path:
                        sys.path.insert(0, str(path.parent))
                    break
        
        if not psx_module_found:
            print("Could not locate PSX module to patch")
            return False
        
        # Now patch the module by monkey patching
        # Import the modules we need to patch
        from psx import stocks, tickers
        import requests
        
        # Save original functions
        original_requests_get = requests.get
        
        # Create patched version
        def patched_requests_get(url, *args, **kwargs):
            # Add headers if not already present
            if 'headers' not in kwargs:
                kwargs['headers'] = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.psx.com.pk/',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0',
                }
            return original_requests_get(url, *args, **kwargs)
        
        # Apply the patch
        requests.get = patched_requests_get
        print("Successfully patched requests.get with custom headers")
        
        # Reload the psx module to apply changes
        importlib.reload(psx)
        
        return True
    except Exception as e:
        print(f"Error applying patch: {str(e)}")
        return False

# Apply the patch when module is imported
patch_result = apply_patch() 