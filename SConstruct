import os
import shutil

# Define the build directory
BUILD_DIR = "build"

# Clean the build directory
def clean(target, source, env):
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    print(f"Cleaned build directory: {BUILD_DIR}")

# Custom build command (e.g., copying files or running scripts)
def build(target, source, env):
    os.makedirs(BUILD_DIR, exist_ok=True)
    # Example: Copy Python source files to the build directory
    shutil.copytree("src/flauto", os.path.join(BUILD_DIR, "flauto"), dirs_exist_ok=True)
    print(f"Built project in {BUILD_DIR}")

# Add the custom commands to SCons
env = Environment()
env.Command("clean", None, clean)
env.Command("build", None, build)

# Default build target
Default("build")
