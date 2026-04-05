"""
Program Mutator: Uses an LLM to mutate problem generation programs.

Supports two mutation types (from Evol-Instruct):
  - In-depth: Make the generated problems harder (add constraints, increase degree, etc.)
  - In-breadth: Create a completely different type of problem generator
"""

import re
import json
from typing import Optional
from .program import ProblemProgram


# Mutation prompts
IN_DEPTH_PROMPT = """You are an expert mathematician and Python programmer.

Below is a Python function that generates math problems using inverse construction 
(it creates the answer first, then constructs the problem from the answer).

```python
{source_code}
```

Your task: Modify this function to generate HARDER problems while keeping the 
inverse construction property (answer must still be mathematically guaranteed correct).

You can make it harder by:
- Increasing the degree or number of variables
- Adding constraints (e.g., require integer solutions, add modular arithmetic)
- Combining multiple concepts
- Increasing the number of reasoning steps needed

IMPORTANT RULES:
1. The function MUST be named `generate` and take a single `seed` argument
2. It MUST return a tuple (problem_text: str, answer: str)
3. The answer MUST be constructed FIRST, then the problem built from it
4. Use only standard library + sympy (if needed)
5. The function must be self-contained (no external dependencies)

Return ONLY the Python code, no explanations. Start with `def generate(seed):` directly.
"""

IN_BREADTH_PROMPT = """You are an expert mathematician and Python programmer.

Below is a Python function that generates math problems using inverse construction:

```python
{source_code}
```

Your task: Create a COMPLETELY DIFFERENT type of math problem generator, 
inspired by but NOT similar to the above. The new generator should:
- Cover a different mathematical topic (e.g., if the above does algebra, try number theory, 
  combinatorics, geometry, or probability)
- Still use inverse construction (create the answer first, build the problem from it)
- Generate problems that require different reasoning skills

IMPORTANT RULES:
1. The function MUST be named `generate` and take a single `seed` argument
2. It MUST return a tuple (problem_text: str, answer: str)  
3. The answer MUST be constructed FIRST, then the problem built from it
4. Use only standard library + sympy (if needed)
5. The function must be self-contained

Return ONLY the Python code, no explanations. Start with `def generate(seed):` directly.
"""


class ProgramMutator:
    """
    LLM-based mutator for problem generation programs.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_vllm: bool = True,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        in_depth_ratio: float = 0.7,  # 70% in-depth, 30% in-breadth
    ):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.in_depth_ratio = in_depth_ratio
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            if self.use_vllm:
                from vllm import LLM
                self._llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    max_model_len=4096,
                )
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._llm = {
                    "model": AutoModelForCausalLM.from_pretrained(
                        self.model_name, torch_dtype="auto", device_map="auto"
                    ),
                    "tokenizer": AutoTokenizer.from_pretrained(self.model_name),
                }
        return self._llm

    def mutate(
        self,
        parent: ProblemProgram,
        mutation_type: str = "auto",
    ) -> Optional[ProblemProgram]:
        """
        Mutate a parent program to create a child program.
        
        Args:
            parent: Parent program to mutate
            mutation_type: "in_depth", "in_breadth", or "auto" (random choice)
        
        Returns:
            New ProblemProgram or None if mutation fails
        """
        import random

        if mutation_type == "auto":
            mutation_type = (
                "in_depth" if random.random() < self.in_depth_ratio else "in_breadth"
            )

        if mutation_type == "in_depth":
            prompt = IN_DEPTH_PROMPT.format(source_code=parent.source_code)
        else:
            prompt = IN_BREADTH_PROMPT.format(source_code=parent.source_code)

        # Generate mutation
        response = self._generate_response(prompt)
        if response is None:
            return None

        # Extract code from response
        source_code = self._extract_code(response)
        if source_code is None:
            return None

        # Validate basic structure
        if "def generate(seed" not in source_code:
            return None

        child = ProblemProgram(
            source_code=source_code,
            parent_id=parent.program_id,
            generation=parent.generation + 1,
            metadata={
                "mutation_type": mutation_type,
                "parent_generation": parent.generation,
            },
        )

        # Quick smoke test: try to execute with a test seed
        test_instance = child.execute(seed=42, timeout=5.0)
        if test_instance is None:
            return None

        return child

    def mutate_batch(
        self,
        parents: list[ProblemProgram],
        mutation_types: Optional[list[str]] = None,
    ) -> list[Optional[ProblemProgram]]:
        """Mutate a batch of parent programs."""
        if mutation_types is None:
            mutation_types = ["auto"] * len(parents)

        results = []
        for parent, mtype in zip(parents, mutation_types):
            child = self.mutate(parent, mtype)
            results.append(child)

        return results

    def _generate_response(self, prompt: str) -> Optional[str]:
        """Generate a response from the LLM."""
        try:
            if self.use_vllm:
                from vllm import SamplingParams
                params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                outputs = self.llm.generate([prompt], params)
                return outputs[0].outputs[0].text
            else:
                model = self.llm["model"]
                tokenizer = self.llm["tokenizer"]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                )
                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                return response
        except Exception as e:
            print(f"[Mutator] Error generating response: {e}")
            return None

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Try to find code block
        code_match = re.search(
            r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL
        )
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Try to find def generate directly
            gen_match = re.search(
                r"(def generate\(seed.*?)(?:\n\n|\Z)", response, re.DOTALL
            )
            if gen_match:
                code = gen_match.group(1).strip()
            else:
                code = response.strip()

        # Basic validation
        if "def generate" not in code:
            return None

        # Remove any non-code content before def
        lines = code.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("def generate") or line.strip().startswith("import") or line.strip().startswith("from"):
                start_idx = i
                break

        code = "\n".join(lines[start_idx:])
        return code if code.strip() else None
