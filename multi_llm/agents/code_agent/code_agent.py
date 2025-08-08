import re
import json
from typing import Dict, Any, Optional, Tuple
from multi_llm.agents.llm_agents.claude import handle
from multi_llm.agents.code_agent.code_agent_prompt import SYSTEM_PROMPT, USER_PROMPT_ERROR, USER_PROMPT


class CodeAgent:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_iterations: int = 5):
        """
        Initialize Code Agent

        Args:
            model: Claude model to use
            max_iterations: Maximum number of iterations for code refinement
        """
        self.model = model
        self.max_iterations = max_iterations
        self.conversation_history = []

        self.system_prompt = SYSTEM_PROMPT

    def extract_code(self, response: str) -> Optional[str]:
        """
        Extract Python code from LLM response

        Args:
            response: LLM response text

        Returns:
            Extracted Python code or None if no code found
        """
        # Look for code blocks marked with ```python
        python_code_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(python_code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback: look for any code blocks
        code_pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        return None

    def generate_code(self, task: str, error_feedback: Optional[str] = None) -> str:
        """
        Generate Python code for the given task

        Args:
            task: Description of the task to solve
            error_feedback: Previous error feedback for code refinement

        Returns:
            Generated Python code
        """
        if error_feedback:
            prompt = USER_PROMPT_ERROR.format(task=task, error_feedback=error_feedback)
        else:
            prompt = USER_PROMPT.format(task=task)

        print(f"Generating code for task: {task[:100]}...")
        if error_feedback:
            print(f"Incorporating error feedback: {error_feedback[:100]}...")

        response = handle(prompt, self.system_prompt, self.model)

        # Store conversation history
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "error_feedback": error_feedback
        })

        return response

    def solve_task(self, task: str, code_tool) -> Dict[str, Any]:
        """
        Solve a task by generating and iteratively refining code

        Args:
            task: Description of the task to solve
            code_tool: CodeExecutionTool instance for running code

        Returns:
            Dictionary containing final result, code, and execution history
        """
        print(f"Starting task: {task}")
        print("=" * 50)

        execution_history = []
        current_code = None
        error_feedback = None

        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            print("-" * 30)

            # Generate code
            llm_response = self.generate_code(task, error_feedback)
            current_code = self.extract_code(llm_response)

            if not current_code:
                print("Error: No Python code found in LLM response")
                execution_history.append({
                    "iteration": iteration + 1,
                    "code": None,
                    "success": False,
                    "error": "No code extracted from LLM response",
                    "output": None
                })
                continue

            print(f"Generated code:\n{current_code}\n")

            # Execute code
            execution_result = code_tool.execute_code(current_code)
            execution_history.append({
                "iteration": iteration + 1,
                "code": current_code,
                "success": execution_result["success"],
                "error": execution_result.get("error"),
                "output": execution_result.get("output")
            })

            if execution_result["success"]:
                print(f"Code executed successfully!")
                print(f"Output: {execution_result['output']}")
                print("=" * 50)

                return {
                    "success": True,
                    "final_code": current_code,
                    "final_output": execution_result["output"],
                    "iterations": iteration + 1,
                    "execution_history": execution_history,
                    "conversation_history": self.conversation_history
                }
            else:
                print(f"Execution failed: {execution_result['error']}")
                error_feedback = f"Error: {execution_result['error']}\nCode that failed:\n{current_code}"

        print(f"Failed to solve task after {self.max_iterations} iterations")
        print("=" * 50)

        return {
            "success": False,
            "final_code": current_code,
            "final_output": None,
            "iterations": self.max_iterations,
            "execution_history": execution_history,
            "conversation_history": self.conversation_history,
            "error": "Maximum iterations reached without successful execution"
        }

    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []
        print("Agent conversation history reset")


if __name__ == "__main__":
    # Example usage
    from multi_llm.agents.tools.code_executor import CodeExecutionTool

    agent = CodeAgent()
    tool = CodeExecutionTool()

    # Test with a mathematical problem
    task = """
    To determine the coefficient of \(x^2y^6\) in the expansion of \(\left(\frac{3}{5}x - \frac{y}{2}\right)^8\), we can use the binomial theorem.

The binomial theorem states:
\[
(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k
\]

In this case, \(a = \frac{3}{5}x\), \(b = -\frac{y}{2}\), and \(n = 8\).

We are interested in the term that contains \(x^2y^6\). In the general term of the binomial expansion:
\[
\binom{8}{k} \left(\frac{3}{5}x\right)^{8-k} \left(-\frac{y}{2}\right)^k
\]

To get \(x^2\), we need \(8 - k = 2\), thus \(k = 6\).

Substituting \(k = 6\) into the expression:
\[
\binom{8}{6} \left(\frac{3}{5}x\right)^{8-6} \left(-\frac{y}{2}\right)^6 = \binom{8}{6} \left(\frac{3}{5}x\right)^2 \left(-\frac{y}{2}\right)^6
\]

Now, we will compute each part of this expression.

1. Calculate the binomial coefficient \(\binom{8}{6}\).
2. Compute \(\left(\frac{3}{5}\right)^2\).
3. Compute \(\left(-\frac{y}{2}\right)^6\).
4. Combine everything together to get the coefficient of \(x^2y^6\).

Let's compute these in Python.
```python
from math import comb

# Given values
n = 8
k = 6

# Calculate the binomial coefficient
binom_coeff = comb(n, k)

# Compute (3/5)^2
a_term = (3/5)**2

# Compute (-1/2)^6
b_term = (-1/2)**6

# Combine terms to get the coefficient of x^2y^6
coefficient = binom_coeff * a_term * b_term
print(coefficient)
```
```output
0.1575
```
The coefficient of \(x^2y^6\) in the expansion of \(\left(\frac{3}{5}x - \frac{y}{2}\right)^8\) is \(0.1575\). To express this as a common fraction, we recognize that:

\[ 0.1575 = \frac{1575}{10000} = \frac{63}{400} \]

Thus, the coefficient can be expressed as:

\[
\boxed{\frac{63}{400}}
\]
    """

    result = agent.solve_task(task, tool)

    if result["success"]:
        print(f"Task completed successfully in {result['iterations']} iterations")
        print(f"Final output: {result['final_output']}")
    else:
        print(f"Task failed: {result['error']}")
