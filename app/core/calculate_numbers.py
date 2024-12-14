from typing import List


async def calculate_numbers(numbers: List[int], operation: str):
    if operation == "sum":
        return sum(numbers)
    elif operation == "subtract":
        return numbers[0] - numbers[1]
    elif operation == "multiply":
        return numbers[0] * numbers[1]
    elif operation == "divide":
        return numbers[0] / numbers[1]
