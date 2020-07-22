import random
import win32com.client as wincl
import sys
import re
import time

def math_q(a_range, b_range, tp=("+", "-", "*", "/"), difficulty = "easy",
           negatives = False, num_decimal_digits = 0):
    "Baseline question generator"
    try:
        op = random.choice(tp)
    except:
        op = tp
    assert op in ("+", "-", "*", "/")

    def _rand(rng):
        v = random.randint(*rng)
        if num_decimal_digits > 0:
            v += random.randint(1, 10**num_decimal_digits - 1) / 10**num_decimal_digits
        if negatives and random.random() > 0.5:
                v *= -1
        return v

    if difficulty in ('medium', 'hard'):
        out = []
        for rng in (a_range, b_range):
            v = str(_rand(rng))
            while difficulty == 'medium' and (len(set(v)) <= len(v)//2
                                             or '0' in v
                                             or '1' in v):
                v = str(_rand(rng))
            while difficulty == 'hard' and (len(set(v)) <= len(v)//2
                                            or '0' in v
                                            or '1' in v
                                            or (op in ('*', '/')
                                                and v[-1] in ('0', '1', '5'))
                                            or (op in ('+', '-')
                                                and '0' in v)):
                v = str(_rand(rng))
            out += [float(v)]
        a,b = out
    else:
        a = _rand(*a_range)
        b = _rand(*b_range)
    if random.random()>0.5:
        a,b = b,a
    if op == "-":
        a = a + b
    elif op == "/":
        a = a * b
    ans = eval(f"{a}{op}{b}")
    return a,b,op, ans

def hidden_math(argv, kwargs, num_qs = None, num_lines = 40):
    """Prints a bunch of lines after the question
    so have to keep all the numbers in your head.
    num_qs: Stops after this many q's
    num_lines: how far to print down after question; set to 0 for no printing
    """
    flash_time = 1.8
    num_right = 0
    num_guesses = 0

    def _finished(num_right, num_guesses, t = time.time()):
        t = time.time() - t
        print(f"Total Correct: {num_right}, in {t//60:.0f}' {t%60:.1f}\" ")
        print(f"Average Accuracy: {num_right/num_guesses *100:.1f}%.",
                  f"Average Time: {t/num_right:.2f}\" ")
        return None

    def _prnt(a,b,op):
        if a % 1 == 0:
            a = int(a)
        if b % 1 == 0:
            b = int(b)
        print(f"{a}  {op}  {b}")
        time.sleep(flash_time)
        print("\n"*num_lines)

    while True:
        a, b, op, ans = math_q(*argv, **kwargs)
        g = ans - 1
        _prnt(a,b,op)
        cnt = 1
        while g != ans:
            if cnt %5 == 0:
                _prnt(a,b,op)
            g = input(": ")
            while 'r' in g:
                _prnt(a,b,op)
                g = input(": ")
            try:
                g = float(g)
            except:
                return _finished(num_right, num_guesses)
            num_guesses += 1
            cnt += 1
        print("Correct!\n\n")
        num_right +=1
        if num_qs and num_right == num_qs:
            return _finished(num_right, num_guesses)

# while True:
#     if False: #random.randint(1, 2) > 1:
#         math_q((1000, 10000), (1000, 10000), tp=("+", "-"))
#     else:

hidden_math([(1003, 9999), (3, 9)], {'tp': ("*"),
                                    'difficulty': 'hard',
                                    'negatives': False,
                                    'num_decimal_digits': 0})
# hidden_math([(13, 99), (13, 99)], {'tp': ("*", '/'), 'difficulty': 'hard'}, num_lines = 40)

# hidden_math([(13, 99), (13, 99)], {'tp': ("*"), 'difficulty': 'hard'}, num_qs = 1)
# hidden_math([(13, 99), (13, 99)], {'tp': ("/"), 'difficulty': 'hard'}, num_qs = 3)

    #%%
def add_aloud(numa, numb):
    "Given the number of digits in the first <= number of digits of second digit to be based asks addition on them"
    speak = wincl.Dispatch("SAPI.SpVoice")
    print(numa, numb)
    while True:
        a = random.randint(3*(10**numa),8*(10**numa))
        b = random.randint(8*(10**(numb-1)), 88*(10**(numb-1)))
        ans = a+b
        g = 0
        if a%7 <= 3:
            b,a = a,b
        speak.Speak(f'{a}, + {b}')
        cnt = 0
        while g!= ans:
            g = input(": ")
            if 'r' in g:
                speak.speak(f'{a}, + {b}')
            else:
                g = int(g)
            cnt += 1
            if cnt % 5 == 0:
                print(a,b)
        print(f"Correct!, {a} + {b} = {ans}\n")

#5x5 grid which flashes blue circles; have to click the cells which held blue
#click arrow which changed direction

#hit 1/0 if coloured circles on left or on right match

#hit 1/0 if center arrow of 5 points left vs. right (and line appears at different heights)

if __name__ == '__main__':
    try:
        digits = list(map(lambda i: int(re.findall(r'\d+', i)[0]), sys.argv[1:]))
        add_aloud(*digits)
    except IndexError:
        add_aloud(3,4)
