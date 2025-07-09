from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(a: float, b: float, operation: str) -> float | str:
    """Useful for performing basic arithmetic calculations with numbers.

    Supported operations: 'add', 'subtract', 'multiply', 'divide', 'floor_divide', 'modulus', 'exponent'

    Args:
        a: First Number
        b: Second Number
        operation: Type of arithmetic operation to perform
    Returns:
        Result of the operation (float or int), or an error message if invalid
    """
    operation = operation.lower().strip()

    if operation == 'add':
        return f"Addition of {a} and {b} would be {a+b}"
    elif operation == "subtract":
        return f"Subtraction of {a} and {b} would be {a-b}"
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero not possible"
        return f"Division of {a} and {b} would be {a/b}"
    elif operation == "floor_divide":
        if b == 0:
            return "Error: Division by zero not possible"
        return f"Floor Division of {a} and {b} would be {a//b}"
    elif operation == "modulus":
        if b == 0:
            return "Error: Division by zero not possible"
        return f"Modulus of {a} and {b} would be {a%b}"
    elif operation == "exponent":
        return f"Exponent of {a} and {b} would be {a**b}"
    else:
        return "Error:Invalid operation"

def main():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # If you want use openai api
    # model = ChatOpenAI(temperature=0)

    tools = [calculator]
    agent_executor = create_react_agent(model, tools)

    print("Welcome I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == 'quit':
            break

        print("\nAssistant: ",end="")
        for chunk in agent_executor.stream(
            {"messages" : [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content,end="")
        print()

if __name__ == "__main__":
    main()
