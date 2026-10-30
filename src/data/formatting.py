"""Prompt templates and formatting for math reasoning (8-shot CoT, zero-shot, chat wrappers)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = (
    "You are an expert mathematical problem solver. "
    "Solve the following math word problem step by step, showing all reasoning clearly. "
    "At the end of your response, provide the final numeric answer on a new line prefixed with '#### ', for example:\n"
    "#### 42"
)

# 8 fixed, cleaned chain-of-thought exemplars carved strictly from the training split
FEW_SHOT_EXEMPLARS: List[Dict[str, str]] = [
    {
        "id": "gsm8k_ece6651237295ac0",
        "question": (
            "Mr. Manuel is a campsite manager who's been tasked with checking the number of "
            "tents set up in the recreation area. On a particular day, he counted 100 tents in the "
            "northernmost part of the campsite and twice that number on the east side of the grounds. "
            "The number of tents at the center of the camp was four times the number of tents in the "
            "northernmost part of the campsite. If he also counted 200 tents in the southern part of the "
            "campsite, what is the total number of tents in the recreation area?"
        ),
        "solution": (
            "On the eastern part of the campsite, Mr Manuel counted 2*100 = 200 tents.\n"
            "The total number of tents in the eastern and the northern part of the campgrounds is 200+100 = 300.\n"
            "There are four times as many tents as the northernmost part in the central part of the campsite, "
            "which means there are 4*100 = 400 tents in the central part of the camp.\n"
            "The total number of tents in the three parts of the campsite is 400+300 = 700.\n"
            "If you add the number of tents Mr. Manuel counted at the southern part of the campsite, you get "
            "700+200 = 900 tents on the whole campsite.\n"
            "#### 900"
        ),
    },
    {
        "id": "gsm8k_1f43ca9b2fea6806",
        "question": (
            "Brenda weighs 10 pounds more than 3 times Mel's weight. If Brenda weighs 220 pounds, "
            "what is Mel's weight?"
        ),
        "solution": (
            "Let Mel's weight be m.\n"
            "Brenda's weight is 3*m + 10 = 220 pounds.\n"
            "Thus 3m = 220 - 10 = 210 pounds.\n"
            "Therefore Mel's weight is m = 210/3 = 70 pounds.\n"
            "#### 70"
        ),
    },
    {
        "id": "gsm8k_93867aaf09b86894",
        "question": (
            "Three snails raced across a rain-soaked sidewalk. The first snail raced at a speed of 2 feet per minute. "
            "The second snail raced at twice the speed of the first snail. And the third snail traveled at five times "
            "the rate of speed as the second snail. If it took the first snail 20 minutes to race all the way up the "
            "sidewalk, how long, in minutes, did it take for the third snail to race all the way up the sidewalk?"
        ),
        "solution": (
            "If the first snail traveled the distance in 20 minutes, then the second snail, traveling at twice the "
            "rate of speed, would take 20/2 = 10 minutes to complete the race.\n"
            "If the third snail raced at five times the speed of the second snail, then it would take the third snail "
            "10/5 = 2 minutes to race all the way up the sidewalk.\n"
            "#### 2"
        ),
    },
    {
        "id": "gsm8k_5658c85502817b56",
        "question": (
            "Dan is digging graves for a family killed by COVID-19. If it takes Dan 3 hours to dig an adult grave and "
            "2 hours to dig a child's grave, how long does it take him to dig 5 adult graves and one child's grave?"
        ),
        "solution": (
            "First find the total amount of time Dan spends digging adult graves: 3 hours/grave * 5 graves = 15 hours.\n"
            "Then add the time to dig the child's grave to find the total time he spends digging: 15 hours + 2 hours = 17 hours.\n"
            "#### 17"
        ),
    },
    {
        "id": "gsm8k_1ac0cc44f9877815",
        "question": (
            "A farmer has 46 chickens. Each chicken gives him 6 eggs a week. If he sells a dozen eggs for $3, "
            "how much money would he make in 8 weeks?"
        ),
        "solution": (
            "The farmer gets 46 x 6 = 276 eggs a week.\n"
            "That gives him 276/12 = 23 dozen eggs.\n"
            "In one week, he makes 23 x 3 = $69.\n"
            "In 8 weeks, he makes $69 x 8 = $552.\n"
            "#### 552"
        ),
    },
    {
        "id": "gsm8k_4e2316a74bdf825a",
        "question": (
            "The company's data entry team had 5 employees working on a large project. Rudy types 64 words per minute, "
            "Joyce types 76 words per minute, Gladys types 91 words per minute, Lisa types 80 words per minute and "
            "Mike types 89 words per minute. What is the team's average typed words per minute?"
        ),
        "solution": (
            "Rudy types 64, Joyce types 76, Gladys types 91, Lisa types 80 and Mike types 89, so 64+76+91+80+89 = 400.\n"
            "There are 5 team members, so they type at an average speed of 400/5 = 80 words per minute.\n"
            "#### 80"
        ),
    },
    {
        "id": "gsm8k_610007d0c44496c3",
        "question": (
            "A cooler is filled with 24 cans of cherry soda and orange pop. If there are twice as many cans of orange "
            "pop as there are of cherry soda, how many cherry sodas are there?"
        ),
        "solution": (
            "There are C cans of cherry soda in the cooler.\n"
            "There are twice as many cans of orange pop, so there are 2C cans of orange pop in the cooler.\n"
            "In all, there are C + 2C = 3C = 24 cans in the cooler.\n"
            "Therefore, there are C = 24 / 3 = 8 cans of cherry soda in the cooler.\n"
            "#### 8"
        ),
    },
    {
        "id": "gsm8k_2726a3ff94ccf170",
        "question": (
            "Bill is trying to count the toddlers at his daycare, but they keep running around. He double-counts 8 "
            "toddlers and misses 3 who are hiding. If Bill thinks he counted 26 toddlers, how many are there really?"
        ),
        "solution": (
            "First subtract the 8 toddlers Bill double-counted from his count: 26 toddlers - 8 toddlers = 18 toddlers.\n"
            "Then add the 3 toddlers he missed to find the real count: 18 toddlers + 3 toddlers = 21 toddlers.\n"
            "#### 21"
        ),
    },
]


def format_few_shot_prompt(question: str, num_shots: int = 8) -> str:
    """Format an 8-shot chain-of-thought prompt for base models without instruct tuning."""
    shots = FEW_SHOT_EXEMPLARS[:num_shots]
    blocks = []
    for ex in shots:
        blocks.append(f"Question: {ex['question']}\nAnswer: {ex['solution']}")

    blocks.append(f"Question: {question.strip()}\nAnswer:")
    return "\n\n".join(blocks)


def format_zero_shot_prompt(question: str) -> str:
    """Format a standard zero-shot prompt with chain-of-thought elicitation."""
    return (
        f"Question: {question.strip()}\n"
        "Answer: Let's think step by step.\n"
    )


def format_chat_messages(
    question: str,
    system_prompt: Optional[str] = None,
    few_shot: bool = False,
    num_shots: int = 8,
) -> List[Dict[str, str]]:
    """Format prompt as OpenAI-compatible chat messages for instruct / RL models."""
    sys_content = system_prompt or SYSTEM_PROMPT
    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_content}]

    if few_shot:
        shots = FEW_SHOT_EXEMPLARS[:num_shots]
        for ex in shots:
            messages.append({"role": "user", "content": ex["question"]})
            messages.append({"role": "assistant", "content": ex["solution"]})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def apply_chat_template(
    messages: List[Dict[str, str]], tokenizer: Optional[Any] = None
) -> str:
    """Render chat messages using tokenizer.apply_chat_template if available, or fallback."""
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback chat format
    rendered_parts: List[str] = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        rendered_parts.append(f"<|{role}|>\n{content}")
    rendered_parts.append("<|Assistant|>\n")
    return "\n\n".join(rendered_parts)
