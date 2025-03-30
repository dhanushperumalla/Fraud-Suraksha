import os
import shutil
from pathlib import Path
import sys

def reset_chroma_db():
    """Reset the Chroma database directory to fix connection issues."""
    chroma_dir = "chroma_db"
    
    print("Fraud Suraksha - Database Reset Utility")
    print("=======================================")
    
    if os.path.exists(chroma_dir):
        print(f"Found Chroma database at: {chroma_dir}")
        print("Removing database files...")
        
        try:
            shutil.rmtree(chroma_dir)
            print("✅ Database files removed successfully.")
        except Exception as e:
            print(f"❌ Error removing database: {str(e)}")
            return False
    else:
        print("No existing database found.")
    
    # Create a fresh directory
    try:
        os.makedirs(chroma_dir, exist_ok=True)
        print("✅ Created fresh database directory.")
        print("Database reset complete! You can now run the app again.")
        return True
    except Exception as e:
        print(f"❌ Error creating directory: {str(e)}")
        return False

if __name__ == "__main__":
    success = reset_chroma_db()
    sys.exit(0 if success else 1) 