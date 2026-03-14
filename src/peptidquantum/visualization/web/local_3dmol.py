"""Local 3Dmol.js bundle management for offline/corporate environments"""
from __future__ import annotations

import urllib.request
from pathlib import Path
import shutil


class Local3DMolManager:
    """Manage local 3Dmol.js bundle for offline use"""
    
    THREEMOL_CDN_URL = "https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"
    THREEMOL_VERSION = "2.0.4"
    
    def __init__(self, bundle_dir: str | Path = "assets/3dmol"):
        """
        Initialize local bundle manager
        
        Args:
            bundle_dir: Directory to store local 3Dmol.js bundle
        """
        self.bundle_dir = Path(bundle_dir)
        self.bundle_file = self.bundle_dir / "3Dmol-min.js"
    
    def download_bundle(self, force: bool = False) -> bool:
        """
        Download 3Dmol.js bundle from CDN
        
        Args:
            force: Force re-download even if file exists
            
        Returns:
            True if successful, False otherwise
        """
        if self.bundle_file.exists() and not force:
            print(f"3Dmol.js bundle already exists: {self.bundle_file}")
            return True
        
        print(f"Downloading 3Dmol.js from {self.THREEMOL_CDN_URL}...")
        
        try:
            self.bundle_dir.mkdir(parents=True, exist_ok=True)
            
            urllib.request.urlretrieve(
                self.THREEMOL_CDN_URL,
                self.bundle_file
            )
            
            print(f"✓ 3Dmol.js bundle downloaded to: {self.bundle_file}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download 3Dmol.js: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if local bundle is available"""
        return self.bundle_file.exists()
    
    def get_bundle_path(self) -> Path:
        """Get path to local bundle"""
        return self.bundle_file
    
    def get_script_tag(self, use_local: bool = False, relative_path: bool = True) -> str:
        """
        Get HTML script tag for 3Dmol.js
        
        Args:
            use_local: Use local bundle instead of CDN
            relative_path: Use relative path for local bundle
            
        Returns:
            HTML script tag
        """
        if use_local and self.is_available():
            if relative_path:
                # Relative path from report location
                path = f"assets/3dmol/3Dmol-min.js"
            else:
                # Absolute path
                path = str(self.bundle_file.absolute())
            
            return f'<script src="{path}"></script>'
        else:
            # Use CDN
            return f'<script src="{self.THREEMOL_CDN_URL}"></script>'
    
    def embed_in_html(self) -> str:
        """
        Get embedded 3Dmol.js code for fully portable HTML
        
        Returns:
            Embedded JavaScript code
        """
        if not self.is_available():
            return ""
        
        with open(self.bundle_file, 'r', encoding='utf-8') as f:
            js_code = f.read()
        
        return f"<script>\n{js_code}\n</script>"
    
    def copy_to_output(self, output_dir: str | Path) -> bool:
        """
        Copy bundle to output directory for portable reports
        
        Args:
            output_dir: Output directory (will create assets/3dmol subdirectory)
            
        Returns:
            True if successful
        """
        if not self.is_available():
            print("Local bundle not available. Download first.")
            return False
        
        output_dir = Path(output_dir)
        target_dir = output_dir / "assets" / "3dmol"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = target_dir / "3Dmol-min.js"
        shutil.copy(self.bundle_file, target_file)
        
        print(f"✓ 3Dmol.js copied to: {target_file}")
        return True


def setup_local_3dmol(bundle_dir: str | Path = "assets/3dmol") -> Local3DMolManager:
    """
    Setup local 3Dmol.js bundle
    
    Args:
        bundle_dir: Directory to store bundle
        
    Returns:
        Local3DMolManager instance
    """
    manager = Local3DMolManager(bundle_dir)
    
    if not manager.is_available():
        print("Local 3Dmol.js bundle not found. Downloading...")
        manager.download_bundle()
    else:
        print(f"✓ Local 3Dmol.js bundle available: {manager.get_bundle_path()}")
    
    return manager


if __name__ == "__main__":
    # CLI for downloading bundle
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage local 3Dmol.js bundle")
    parser.add_argument("--download", action="store_true", help="Download 3Dmol.js bundle")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--dir", type=Path, default="assets/3dmol", help="Bundle directory")
    
    args = parser.parse_args()
    
    manager = Local3DMolManager(args.dir)
    
    if args.download:
        manager.download_bundle(force=args.force)
    else:
        if manager.is_available():
            print(f"✓ 3Dmol.js bundle available: {manager.get_bundle_path()}")
        else:
            print("✗ 3Dmol.js bundle not found")
            print("Run with --download to download")
