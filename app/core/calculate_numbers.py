from typing import List


async def calculate_numbers(numbers: List[int], operation: str):
    if operation == "+":
        return sum(numbers)
    elif operation == "-":
        return numbers[0] - numbers[1]
    elif operation == "*":
        return numbers[0] * numbers[1]
    elif operation == "/":
        return numbers[0] / numbers[1]
