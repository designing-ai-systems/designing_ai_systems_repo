"""
Generate Python code from Protocol Buffer definitions.

Run this script to generate Python stubs from .proto files.
"""

import subprocess
import sys
from pathlib import Path


def generate_proto_code():
    """Generate Python code from .proto files."""
    # Get the directory containing this script
    proto_dir = Path(__file__).parent

    # Output directory for generated code
    output_dir = proto_dir

    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))

    if not proto_files:
        print("No .proto files found")
        return

    print(f"Found {len(proto_files)} .proto file(s)")

    # Generate Python code for each .proto file
    for proto_file in proto_files:
        print(f"Generating code for {proto_file.name}...")

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grpc_tools.protoc",
                    f"--proto_path={proto_dir}",
                    f"--python_out={output_dir}",
                    f"--grpc_python_out={output_dir}",
                    str(proto_file),
                ],
                check=True,
            )

            # Fix import in generated grpc file
            grpc_file = output_dir / f"{proto_file.stem}_pb2_grpc.py"
            if grpc_file.exists():
                content = grpc_file.read_text()
                # Replace relative import with absolute import
                content = content.replace(
                    f"import {proto_file.stem}_pb2 as {proto_file.stem}__pb2",
                    f"from proto import {proto_file.stem}_pb2 as {proto_file.stem}__pb2",
                )
                grpc_file.write_text(content)

            print(f"✓ Generated code for {proto_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error generating code for {proto_file.name}: {e}")
            sys.exit(1)

    print("\n✓ All Protocol Buffer code generated successfully!")


if __name__ == "__main__":
    generate_proto_code()
