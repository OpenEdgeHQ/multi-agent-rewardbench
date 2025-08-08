import sys
import io
import contextlib
import traceback
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
import threading
import time


class CodeExecutionTool:
    def __init__(self, timeout: int = 30, max_output_length: int = 10000):
        """
        Initialize Code Execution Tool

        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum length of captured output
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.execution_count = 0

    def execute_code(self, code: str, method: str = "exec") -> Dict[str, Any]:
        """
        Execute Python code and return results

        Args:
            code: Python code to execute
            method: Execution method ("exec" for in-process, "subprocess" for isolated)

        Returns:
            Dictionary containing execution results
        """
        self.execution_count += 1
        print(f"Executing code (attempt #{self.execution_count})...")

        if method == "subprocess":
            return self._execute_subprocess(code)
        else:
            return self._execute_in_process(code)

    def _execute_in_process(self, code: str) -> Dict[str, Any]:
        """
        Execute code in the current Python process with output capture

        Args:
            code: Python code to execute

        Returns:
            Dictionary containing execution results
        """
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        execution_result = {
            "success": False,
            "output": None,
            "error": None,
            "execution_time": None
        }

        try:
            # Redirect output streams
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            start_time = time.time()

            # Create execution namespace with commonly used imports
            namespace = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
            }

            # Pre-import commonly used modules
            common_imports = """
import math
import numpy as np
import scipy
from scipy import optimize, integrate, stats
import sympy as sp
from sympy import symbols, diff, integrate as sp_integrate, solve, simplify, expand, factor
import matplotlib.pyplot as plt
import pandas as pd
import json
import re
import itertools
from collections import Counter, defaultdict
from fractions import Fraction
import decimal
from decimal import Decimal, getcontext
"""

            # Execute common imports first
            exec(common_imports, namespace)

            # Execute the user code
            exec(code, namespace)

            execution_time = time.time() - start_time

            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            # Combine outputs
            combined_output = ""
            if stdout_output:
                combined_output += stdout_output
            if stderr_output:
                if combined_output:
                    combined_output += "\nSTDERR:\n"
                combined_output += stderr_output

            # Truncate if too long
            if len(combined_output) > self.max_output_length:
                combined_output = combined_output[:self.max_output_length] + "\n... (output truncated)"

            execution_result.update({
                "success": True,
                "output": combined_output if combined_output else "Code executed successfully (no output)",
                "execution_time": execution_time
            })

            print(f"Code executed successfully in {execution_time:.3f} seconds")

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"{type(e).__name__}: {str(e)}"

            # Get traceback
            tb_str = traceback.format_exc()

            execution_result.update({
                "success": False,
                "error": error_message,
                "traceback": tb_str,
                "execution_time": execution_time
            })

            print(f"Code execution failed: {error_message}")

        finally:
            # Restore output streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return execution_result

    def _execute_subprocess(self, code: str) -> Dict[str, Any]:
        """
        Execute code in a separate subprocess for better isolation

        Args:
            code: Python code to execute

        Returns:
            Dictionary containing execution results
        """
        execution_result = {
            "success": False,
            "output": None,
            "error": None,
            "execution_time": None
        }

        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            # Add common imports at the beginning
            full_code = """
import math
import numpy as np
import scipy
from scipy import optimize, integrate, stats
import sympy as sp
from sympy import symbols, diff, integrate as sp_integrate, solve, simplify, expand, factor
import matplotlib.pyplot as plt
import pandas as pd
import json
import re
import itertools
from collections import Counter, defaultdict
from fractions import Fraction
import decimal
from decimal import Decimal, getcontext

""" + code

            temp_file.write(full_code)
            temp_file_path = temp_file.name

        try:
            start_time = time.time()

            # Execute the code in subprocess
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                # Success
                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"

                # Truncate if too long
                if len(output) > self.max_output_length:
                    output = output[:self.max_output_length] + "\n... (output truncated)"

                execution_result.update({
                    "success": True,
                    "output": output if output else "Code executed successfully (no output)",
                    "execution_time": execution_time
                })

                print(f"Code executed successfully in subprocess in {execution_time:.3f} seconds")

            else:
                # Error
                error_output = result.stderr if result.stderr else result.stdout
                execution_result.update({
                    "success": False,
                    "error": f"Process exited with code {result.returncode}: {error_output}",
                    "execution_time": execution_time
                })

                print(f"Code execution failed in subprocess: {error_output}")

        except subprocess.TimeoutExpired:
            execution_result.update({
                "success": False,
                "error": f"Code execution timed out after {self.timeout} seconds",
                "execution_time": self.timeout
            })
            print(f"Code execution timed out after {self.timeout} seconds")

        except Exception as e:
            execution_result.update({
                "success": False,
                "error": f"Subprocess execution error: {str(e)}",
                "execution_time": None
            })
            print(f"Subprocess execution error: {str(e)}")

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

        return execution_result

    def install_package(self, package_name: str) -> bool:
        """
        Install a Python package using pip

        Args:
            package_name: Name of the package to install

        Returns:
            True if installation successful, False otherwise
        """
        try:
            print(f"Installing package: {package_name}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout for package installation
            )

            if result.returncode == 0:
                print(f"Successfully installed {package_name}")
                return True
            else:
                print(f"Failed to install {package_name}: {result.stderr}")
                return False

        except Exception as e:
            print(f"Error installing {package_name}: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics

        Returns:
            Dictionary containing execution statistics
        """
        return {
            "total_executions": self.execution_count,
            "timeout_setting": self.timeout,
            "max_output_length": self.max_output_length
        }

    def reset_stats(self):
        """Reset execution statistics"""
        self.execution_count = 0
        print("Execution statistics reset")


if __name__ == "__main__":
    # Example usage
    tool = CodeExecutionTool()

    # Test basic calculation
    test_code1 = """
# Calculate factorial of 10
import math
result = math.factorial(10)
print(f"10! = {result}")
"""

    print("Testing basic calculation:")
    result1 = tool.execute_code(test_code1)
    print(f"Success: {result1['success']}")
    print(f"Output: {result1['output']}")
    print()

    # Test mathematical derivation
    test_code2 = """
# Find derivative and roots of polynomial
import sympy as sp
import numpy as np

# Define variable and function
x = sp.Symbol('x')
f = x**3 + 2*x**2 - 5*x + 3

print(f"Function: f(x) = {f}")

# Calculate derivative
f_prime = sp.diff(f, x)
print(f"Derivative: f'(x) = {f_prime}")

# Evaluate derivative at x = 2
derivative_at_2 = f_prime.subs(x, 2)
print(f"f'(2) = {derivative_at_2}")

# Find roots
roots = sp.solve(f, x)
print(f"Roots of f(x) = 0: {roots}")

# Numerical evaluation of roots
numerical_roots = [float(root.evalf()) for root in roots if root.is_real]
print(f"Numerical roots: {numerical_roots}")
"""

    print("Testing mathematical derivation:")
    result2 = tool.execute_code(test_code2)
    print(f"Success: {result2['success']}")
    print(f"Output: {result2['output']}")
    print()

    # Test error handling
    test_code3 = """
# This will cause an error
result = 1 / 0
print(result)
"""

    print("Testing error handling:")
    result3 = tool.execute_code(test_code3)
